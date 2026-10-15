"""生成核心 —— 供 `web/server.py` 导入。**本文件不含任何界面代码。**

与上游 `LuxTTS` 类的关系
------------------------
上游把「加载模型」写死在 `zipvoice/modeling_utils.py` 的 `load_models_gpu()` /
`load_models_cpu()` 里，而那两处对本地环境有四处不成立（`docs/plans` 的 §0.1 有完整清单）：

1. `load_models_cpu()` 第一行**无条件** `snapshot_download(...)`，把调用方传进来的
   `checkpoints` 直接扔掉 → CPU 模式每次联网重下 1.1 GB；
2. 转写模型在 GPU 路径是**硬编码相对路径**、在 CPU 路径是联网的 `openai/whisper-tiny`；
3. `torch.device(device, 0)` 硬编码 0 号卡；
4. 所有模型路径都按「当前工作目录」解析。

所以这里**自己装配模型**：照抄上游那几十行的语义，但路径全部绝对、转写器可换、
设备可指定。**好处是 `upstream/` 目录保持 100% 原样、不产生任何 diff** ——
升级上游时不会撞冲突，也不需要维护一份打了补丁的副本。

采样与声码器这两段复杂逻辑**没有重复实现**，仍然调用上游函数
（`zipvoice.modeling_utils.generate` / `zipvoice.onnx_modeling.generate_cpu`）。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import asr as asr_mod
from . import config, hardware

log = logging.getLogger("luxtts.core")

_LOCK = threading.Lock()
_STATE: dict = {
    "loaded": False,
    "device": None,
    "model": None,
    "vocos": None,
    "tokenizer": None,
    "feature_extractor": None,
    "load_seconds": None,
    "is_cpu_path": False,
}


# --------------------------------------------------------------------------
# 模型装配
# --------------------------------------------------------------------------
def _build(device: torch.device) -> dict:
    from torch.nn.utils import parametrize

    from linacodec.vocoder.vocos import Vocos
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank

    cp = config.CHECKPOINTS
    for need in (cp / "config.json", cp / "tokens.txt", config.VOCODER_DIR / "vocos.bin"):
        if not need.exists():
            raise FileNotFoundError(f"模型文件缺失：{need}")

    cfg = json.loads((cp / "config.json").read_text(encoding="utf-8"))
    tokenizer = EmiliaTokenizer(token_file=str(cp / "tokens.txt"))
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    feature_extractor = VocosFbank()

    is_cpu = device.type == "cpu"
    if is_cpu:
        # CPU 走 ONNX。这里**沿用上游选的文件**（非 int8 那两个），
        # 目录里的 *_int8.onnx 更小更快，但属未经验证的替换，先不动。
        from zipvoice.onnx_modeling import OnnxModel

        model = OnnxModel(
            str(cp / "text_encoder.onnx"),
            str(cp / "fm_decoder.onnx"),
            num_thread=hardware.num_threads(),
        )
        log.info("已装配 ONNX(CPU) 模型，线程 %d", hardware.num_threads())
    else:
        model = ZipVoiceDistill(**cfg["model"], **tokenizer_config)
        load_checkpoint(filename=str(cp / "model.pt"), model=model, strict=True)
        model = model.to(device).eval()
        log.info("已装配 PyTorch 模型于 %s", device)

    vocos = Vocos.from_hparams(str(config.VOCODER_DIR / "config.yaml"))
    # 上游就是先摘 parametrization 再 load_state_dict，顺序别换。
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    if is_cpu:
        vocos = vocos.eval()
        state = torch.load(str(config.VOCODER_DIR / "vocos.bin"), map_location="cpu")
    else:
        vocos = vocos.to(device)
        state = torch.load(str(config.VOCODER_DIR / "vocos.bin"), map_location=device)
    vocos.load_state_dict(state)
    vocos.freq_range = 12000          # 上游 LuxTTS.__init__ 里设的，别漏

    return {
        "device": device,
        "model": model,
        "vocos": vocos,
        "tokenizer": tokenizer,
        "feature_extractor": feature_extractor,
        "is_cpu_path": is_cpu,
    }


def load_model(force_reload: bool = False) -> dict:
    """装载模型（进程内单例）。**不在服务启动时调用** —— 界面要能秒开、可反复验证。"""
    with _LOCK:
        if _STATE["loaded"] and not force_reload:
            return _STATE
        t0 = time.perf_counter()
        built = _build(hardware.resolve_device())
        _STATE.update(built)
        _STATE["loaded"] = True
        _STATE["load_seconds"] = round(time.perf_counter() - t0, 2)
        log.info("模型加载完成，用时 %.2f s", _STATE["load_seconds"])
        return _STATE


def is_loaded() -> bool:
    return bool(_STATE["loaded"])


def unload() -> None:
    with _LOCK:
        _STATE.update(
            loaded=False, model=None, vocos=None, tokenizer=None,
            feature_extractor=None, device=None, load_seconds=None, is_cpu_path=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def device_info() -> dict:
    """界面顶部信息条 + 排障用。未加载模型时也会给出「如果加载会用什么」。"""
    prof = hardware.active_profile()
    return {
        "loaded": is_loaded(),
        "device": str(_STATE["device"]) if _STATE["device"] is not None else prof["device_str"],
        "label": prof["label"],
        "gpu": prof["gpu"],
        "load_seconds": _STATE["load_seconds"],
        "cpu_path": bool(_STATE["is_cpu_path"]),
        "torch": prof["torch"],
        "cuda_built": prof["cuda_built"],
        "threads": prof["threads"],
        "asr_engine": config.ASR_ENGINE,
        "asr_enabled": config.ASR_ENABLED,
        "asr_loaded": asr_mod.is_loaded(),
        "asr_reason": asr_mod.unavailable_reason(),
    }


def active_profile() -> dict:
    return hardware.active_profile()


# --------------------------------------------------------------------------
# 参考音频编码
# --------------------------------------------------------------------------
def _require_text(ref_text: str | None, reason: str | None) -> str:
    """拿不到参考文本时**直接报错要求手填**，不静默降级。

    上游在这里会退回内置 ASR（可能触发联网下载）；本项目不做兜底，
    因为「以为是克隆、其实参考文本是空的」比直接报错难查得多。
    """
    text = (ref_text or "").strip()
    if text:
        return text
    hint = f"（自动转写不可用：{reason}）" if reason else ""
    raise RuntimeError(
        "拿不到参考音频的文本，无法建立音色。请在界面上手工填写「参考文本」再试。" + hint
    )


def _speech_rate(text: str, seconds: float) -> float:
    """参考文本的「字/秒」。中文字 + 英文单词都算一个字。"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    words = sum(1 for w in text.split() if w)
    return (cjk + words) / max(0.1, seconds)


def _check_text_matches_audio(text: str, frames: int, sample_rate: int = config.SR_24K) -> None:
    """参考文本必须和特征那一段音频**对应**，否则语速会错得离谱。

    上游的目标时长是这么推的（`zipvoice/models/zipvoice.py`）::

        目标帧数 = 参考帧数 / 参考token数 * 目标token数 / speed

    注意分子是**参考帧数**（只含 `ref_duration` 那一段），分母却是**整段文本的
    token 数**。两者一旦不对应，比例就整体偏移 —— 文本越长、语速越快，且**没有任何报错**。

    实测：20 秒音频、界面按默认 `ref_duration=5` 建缓存，但转写时没截断、
    把整段 112 字都存成了参考文本 → 同一句话输出 **26.7 字/秒**（正常约 6）。
    因为 112/28 ≈ 4，时长正好被压成四分之一。

    所以这里设一道护栏：按「文本字数 ÷ 特征时长」算出隐含语速，明显超出人类
    说话范围就报错，**宁可建音色失败，也不要悄悄生成一个语速错乱的音色**。
    """
    # 特征是mel 帧，固定 25 帧/秒（VocosFbank 的 hop）—— **与采样率无关**，
    # 之前写成 frames / (sample_rate/25) 会把 5 秒算成 0.13 秒，满盘误报。
    seconds = frames / 25.0
    rate = _speech_rate(text, seconds)
    if rate > 14.0 or rate < 0.8:
        raise RuntimeError(
            f"参考文本与参考音频对不上：{rate:.1f} 字/秒（正常说话约 3~8 字/秒）。"
            f"通常是因为转写用了整段音频，而 prompt 只取了前 {seconds:.1f} 秒 —— "
            f"转写与建缓存必须用同一段音频。请在新建音色时先裁剪波形再转写。"
        )


@torch.inference_mode()
def encode_ref(
    audio: str | Path | np.ndarray,
    *,
    duration: float | None = None,
    rms: float | None = None,
    ref_text: str | None = None,
    sample_rate: int | None = None,
) -> dict:
    """把参考音频编码成一个 prompt（可直接喂给 :func:`synthesize`，也可落盘缓存）。

    参数
    ----
    audio       音频文件路径，或 ``(np.ndarray, sample_rate)`` 里的数组。
    duration    只取前 N 秒（``None`` = 用 ``config.DEFAULTS['ref_duration']``；``<=0`` = 整段）。
    rms         响度归一化目标（``None`` = 默认值）。
    ref_text    已知的参考文本。**给了就完全跳过转写** —— 建音色会快一截，
                而且不必依赖转写引擎可用。
    sample_rate 仅当 ``audio`` 是数组时使用。

    返回的 dict 除上游那四个 key（``prompt_tokens`` / ``prompt_features_lens`` /
    ``prompt_features`` / ``prompt_rms``）外，还多带 ``ref_text`` 与 ``ref_seconds``
    两个元信息（供音色库落盘；上游的 ``generate()`` 是按名取参数，多出来的键不影响它）。
    """
    import librosa

    from zipvoice.utils.infer import rms_norm

    st = load_model()
    tokenizer = st["tokenizer"]
    feature_extractor = st["feature_extractor"]
    device = st["device"]

    if duration is None:
        duration = float(config.DEFAULTS["ref_duration"])
    if rms is None:
        rms = float(config.DEFAULTS["rms"])

    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("传入数组时必须给 sample_rate")
        # librosa.load 负责解码与重采样；数组路径自己重采样，保证两条路一致。
        import librosa as _lb

        wav24 = _lb.resample(np.asarray(audio, dtype=np.float32), orig_sr=sample_rate, target_sr=config.SR_24K)
        wav16 = _lb.resample(np.asarray(audio, dtype=np.float32), orig_sr=sample_rate, target_sr=16000)
        if duration and duration > 0:
            wav24 = wav24[: int(duration * config.SR_24K)]
            wav16 = wav16[: int(duration * 16000)]
    else:
        dur = None if not duration or duration <= 0 else float(duration)
        wav24, _ = librosa.load(str(audio), sr=config.SR_24K, duration=dur)
        wav16 = None
        if not (ref_text or "").strip():
            wav16, _ = librosa.load(str(audio), sr=16000, duration=dur)

    text = (ref_text or "").strip()
    if not text:
        transcriber = asr_mod.get_transcriber(device)
        if transcriber is None:
            text = _require_text(None, asr_mod.unavailable_reason())
        else:
            text = (transcriber({"array": wav16, "sampling_rate": 16000}).get("text") or "").strip()
    text = _require_text(text, asr_mod.unavailable_reason())

    ref_seconds = round(len(wav24) / config.SR_24K, 2)

    prompt_wav = torch.from_numpy(np.asarray(wav24, dtype=np.float32)).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, float(rms))
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=config.SR_24K).to(device)
    prompt_features = prompt_features.unsqueeze(0) * 0.1      # feat_scale，上游固定 0.1
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([text])

    # 文本必须与特征那一段对应，否则语速会整体偏移（见 _check_text_matches_audio）
    _check_text_matches_audio(text, prompt_features.size(1), config.SR_24K)

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_features_lens": prompt_features_lens,
        "prompt_features": prompt_features,
        "prompt_rms": prompt_rms,
        "ref_text": text,
        "ref_seconds": ref_seconds,
        "ref_rate": round(_speech_rate(text, ref_seconds), 2),
    }


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
PROMPT_KEYS = ("prompt_tokens", "prompt_features_lens", "prompt_features", "prompt_rms")


def _prompt_only(prompt: dict, device: torch.device) -> tuple:
    """取出四个字段并把张量对齐到模型所在设备。

    **这一步不能省。** 从 ``voices/<名>.pt`` 读回来的缓存是用
    ``map_location="cpu"`` 加载的，而模型在 CUDA 上 —— 不搬设备的话，
    ``prompt_features`` 会在 forward 里抛
    「Expected all tensors to be on the same device」。
    （上游 ``generate()`` 的默认值里有 ``prompt_rms`` 参与标量运算，
    三个张量一起搬最省心。）
    """
    missing = [k for k in PROMPT_KEYS if k not in prompt]
    if missing:
        raise ValueError(f"prompt 缺少字段：{', '.join(missing)}（是不是传了别的结构的缓存？）")
    tokens = prompt["prompt_tokens"]
    moved = []
    for k in PROMPT_KEYS[1:]:
        v = prompt[k]
        moved.append(v.to(device) if torch.is_tensor(v) else v)
    return (tokens, *moved)


def merged_params(overrides: dict | None = None) -> dict:
    """默认参数 + 覆盖值，并做范围校验。界面和历史回填都走这里，保证只有一份真源。"""
    p = dict(config.DEFAULTS)
    for k, v in (overrides or {}).items():
        if k in p and v is not None:
            p[k] = v
    lo, hi = config.PARAM_RANGES["num_steps"]
    p["num_steps"] = int(max(lo, min(hi, int(round(float(p["num_steps"]))))))
    p["return_smooth"] = bool(p["return_smooth"])
    for k in ("rms", "ref_duration", "t_shift", "speed"):
        lo, hi = config.PARAM_RANGES[k]
        p[k] = float(max(lo, min(hi, float(p[k]))))
    return p


def synthesize(text: str, prompt: dict, params: dict | None = None) -> dict:
    """用给定的 prompt 合成 ``text``。

    返回 ``{"wav": np.float32[单声道], "sr": int, "seconds": float, "elapsed": float}``。

    **``sr`` 必须一起返回**：声码器有 48k / 24k 两个头，``return_smooth`` 会切换它们，
    前端若一律按 48k 播，24k 那条会变速。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("合成文本不能为空")

    p = merged_params(params)
    st = load_model()
    device = st["device"]

    if device.type == "cpu":
        from zipvoice.onnx_modeling import generate_cpu as _generate
    else:
        from zipvoice.modeling_utils import generate as _generate

    # return_smooth=True 用 24k 的平滑头（上游的语义：return_48k = not return_smooth）
    st["vocos"].return_48k = not p["return_smooth"]
    sr = config.SR_24K if p["return_smooth"] else config.SR_48K

    tokens, lens, features, prms = _prompt_only(prompt, device)
    t0 = time.perf_counter()
    wav = _generate(
        tokens, lens, features, prms, text,
        st["model"], st["vocos"], st["tokenizer"],
        num_step=int(p["num_steps"]),
        t_shift=float(p["t_shift"]),
        speed=float(p["speed"]),
    )
    elapsed = time.perf_counter() - t0

    audio = wav.cpu().squeeze().numpy().astype(np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return {
        "wav": audio,
        "sr": sr,
        "seconds": round(len(audio) / sr, 2),
        "elapsed": round(elapsed, 2),
        "params": p,
    }


def synthesize_from_audio(
    text: str, audio: str | Path, params: dict | None = None,
    *, ref_text: str | None = None,
) -> dict:
    """一次性合成（不建音色、不缓存 prompt）。界面上「临时试听」用。"""
    p = merged_params(params)
    prompt = encode_ref(
        audio, duration=p["ref_duration"], rms=p["rms"], ref_text=ref_text,
    )
    out = synthesize(text, prompt, p)
    out["ref_text"] = prompt["ref_text"]
    out["ref_seconds"] = prompt["ref_seconds"]
    return out


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------
def save_wav(path: str | Path, wav: np.ndarray, sr: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr)
    return path
