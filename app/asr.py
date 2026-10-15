"""参考音频转写 —— 可插拔。

为什么需要这一层：上游把转写**写死**在 `zipvoice/modeling_utils.py` 的
`load_models_gpu()` / `load_models_cpu()` 里 —— GPU 路径用**硬编码相对路径**
`whisper-base/snapshots/e37978b…`（换个工作目录就找不到），CPU 路径干脆用
`openai/whisper-tiny`（**会联网重下**，本地的 `whisper-base/` 白放着）。
而且上游 `LuxTTS` 是把转写器存成**实例属性**的（`zipvoice/luxvoice.py:34`），
所以换引擎本来就不需要改上游代码 —— 只要喂一个满足同样调用约定的可调用对象。

本模块的契约（与上游 `process_audio()` 的调用方式一致）::

    transcriber({"array": np.ndarray, "sampling_rate": int}) -> {"text": str}

两个引擎：

* ``whisper``     —— 默认。走本地 `whisper-base/`，**不联网**。
* ``sensevoice``  —— SenseVoice Small（sherpa-onnx int8，纯 CPU，0.24 GB）。
  与 OmniVoice Studio 用同一个引擎；权重可从 OmniVoice 项目直接复制，无需下载。
  需要额外装 `sherpa-onnx`，没装时 `unavailable_reason()` 会说清楚原因。

注意（实测）：whisper-base 转中文**会输出繁体**（例：「在這個信息越來越碎片化的時代…」）。
繁体字经 pypinyin 照样能出音素、不影响合成，但界面上「参考文本」是可编辑的，
发现不对可以自己改；想从源头拿到简体就切 `sensevoice`。
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

log = logging.getLogger("luxtts.asr")

_LOCK = threading.Lock()
_CACHE: dict = {}

WHISPER_FALLBACK = "openai/whisper-tiny"   # 本地快照缺失时才联网（上游 CPU 路径的行为）


# --------------------------------------------------------------------------
# 本地 Whisper 目录定位
# --------------------------------------------------------------------------
def resolve_whisper_dir() -> str | None:
    """找到本地 whisper 快照目录，找不到返回 None（由调用方决定是否联网兜底）。

    优先按 `refs/main` 里记的 revision 精确命中，其次取 `snapshots/` 下唯一的那个目录。
    """
    snap = config.WHISPER_DIR / "snapshots"
    if not snap.is_dir():
        return None
    ref = config.WHISPER_DIR / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8", errors="replace").strip()
        cand = snap / rev
        if cand.is_dir():
            return str(cand)
    subs = sorted(p for p in snap.iterdir() if p.is_dir())
    return str(subs[0]) if subs else None


# --------------------------------------------------------------------------
# 引擎实现
# --------------------------------------------------------------------------
def _build_whisper(device) -> object:
    from transformers import pipeline

    local = resolve_whisper_dir()
    model_id = local or WHISPER_FALLBACK
    if local is None:
        log.warning(
            "本地 %s 不存在，回退到联网下载 %s（若想避免联网，请恢复 whisper-base/ 目录）",
            config.WHISPER_DIR, WHISPER_FALLBACK,
        )
    log.info("转写引擎 whisper：%s @ %s", model_id, device)
    pipe = pipeline("automatic-speech-recognition", model=model_id, device=_device_arg(device))
    return _WhisperCaller(pipe)


def _device_arg(device) -> object:
    """transformers 的 device 参数：CPU 传 -1，CUDA 传 cuda:N。"""
    if getattr(device, "type", str(device)) == "cpu":
        return -1
    if getattr(device, "type", None) == "cuda":
        return f"cuda:{device.index or 0}"
    return -1


class _WhisperCaller:
    def __init__(self, pipe):
        self.name = "whisper"
        self._pipe = pipe

    def __call__(self, payload: dict) -> dict:
        audio = np.asarray(payload["array"], dtype=np.float32)
        out = self._pipe({"array": audio, "sampling_rate": int(payload["sampling_rate"])})
        text = (out or {}).get("text", "") or ""
        return {"text": text.strip()}


class _SenseVoiceCaller:
    def __init__(self, recognizer, sample_rate: int = 16000):
        self.name = "sensevoice"
        self._rec = recognizer
        self._sr = sample_rate

    def __call__(self, payload: dict) -> dict:
        audio = np.asarray(payload["array"], dtype=np.float32)
        stream = self._rec.create_stream()
        stream.accept_waveform(self._sr, audio)
        self._rec.decode_stream(stream)
        text = getattr(stream.result, "text", "") or ""
        return {"text": text.strip()}


def _build_sensevoice() -> object:
    import sherpa_onnx

    d = config.SENSEVOICE_DIR
    model, tokens = d / "model.int8.onnx", d / "tokens.txt"
    n = config.ASR_THREADS if config.ASR_THREADS > 0 else 2
    # use_itn=True 才有标点 —— 这是 OmniVoice 项目实测确认的，关掉会输出整段无标点。
    kwargs = dict(model=str(model), tokens=str(tokens), use_itn=True, num_threads=n)
    try:
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(language="auto", **kwargs)
    except TypeError:
        # 某些版本的 from_sense_voice 不接受 language，退回最小参数集。
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(**kwargs)
    log.info("转写引擎 sensevoice：%s", d)
    return _SenseVoiceCaller(rec)


# --------------------------------------------------------------------------
# 可用性
# --------------------------------------------------------------------------
def shift_to_simplified_hint() -> str:
    """给界面用的一句提示（不自动转换文本，只说明现象与出路）。"""
    return "Whisper 转中文可能出繁体（不影响合成）；要简体请把 ASR 引擎切到 sensevoice。"


def unavailable_reason(engine: str | None = None) -> str | None:
    """返回「为什么这个引擎用不了」，能用则返回 None。

    三种原因要分清楚，否则排障时会白找：
    整体关闭 / 缺 python 包 / 缺模型文件。
    """
    engine = (engine or config.ASR_ENGINE).strip().lower()
    if not config.ASR_ENABLED:
        return "自动转写已整体关闭（环境变量 LUXTTS_ASR=0）→ 新建音色时需要手工填写参考文本"
    if engine == "whisper":
        if resolve_whisper_dir() is None:
            return (
                f"本地转写模型缺失：{config.WHISPER_DIR}\\snapshots\\<rev>\\ 不存在"
                "（会退化成联网下载 openai/whisper-tiny）"
            )
        return None
    if engine == "sensevoice":
        import importlib.util

        if importlib.util.find_spec("sherpa_onnx") is None:
            return "缺少 python 包 sherpa-onnx（装：runtime\\python.exe -m pip install sherpa-onnx）"
        d = config.SENSEVOICE_DIR
        missing = [f for f in ("model.int8.onnx", "tokens.txt") if not (d / f).exists()]
        if missing:
            return f"缺少 SenseVoice 模型文件：{d}\\{', '.join(missing)}"
        return None
    return f"未知的转写引擎：{engine!r}（可选 whisper / sensevoice）"


# --------------------------------------------------------------------------
# 单例
# --------------------------------------------------------------------------
def get_transcriber(device=None, force_reload: bool = False):
    """惰性构建转写器（进程内单例）。转写与模型加载互不阻塞，各自加锁。"""
    engine = (config.ASR_ENGINE or "whisper").strip().lower()
    with _LOCK:
        if engine in _CACHE and not force_reload:
            return _CACHE[engine]
        reason = unavailable_reason(engine)
        if reason is not None:
            log.warning("转写引擎 %s 不可用：%s", engine, reason)
            return None
        if device is None:
            from . import hardware

            device = hardware.resolve_device()
        obj = _build_sensevoice() if engine == "sensevoice" else _build_whisper(device)
        _CACHE[engine] = obj
        return obj


def is_loaded(engine: str | None = None) -> bool:
    return (engine or config.ASR_ENGINE) in _CACHE


def unload() -> None:
    with _LOCK:
        _CACHE.clear()
