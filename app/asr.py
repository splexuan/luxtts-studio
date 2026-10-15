"""参考音频转写 —— SenseVoice Small（唯一引擎）。

为什么需要这一层：上游把转写**写死**在 `zipvoice/modeling_utils.py` 的
`load_models_gpu()` / `load_models_cpu()` 里 —— GPU 路径用**硬编码相对路径**
`whisper-base/snapshots/e37978b…`（换个工作目录就找不到），CPU 路径干脆用
`openai/whisper-tiny`（**会联网重下**）。而且上游 `LuxTTS` 是把转写器存成
**实例属性**的（`zipvoice/luxvoice.py:34`），所以换引擎本来就不需要改上游
代码 —— 只要喂一个满足同样调用约定的可调用对象。

本模块的契约（与上游 `process_audio()` 的调用方式一致）::

    transcriber({"array": np.ndarray, "sampling_rate": int}) -> {"text": str}

**为什么是 SenseVoice 而不是 Whisper**（2026-10-15 实测后定死，别再加回 Whisper）：

* Whisper base 转中文**输出繁体**（「在這個信息越來越碎片化的時代…」），
  且慢在**权重装载**——transformers 装载要 9.69 s（冷），转写计算本身反而只要 3.38 s；
* SenseVoice Small 是 sherpa-onnx int8、纯 CPU、229 MB，**冷启秒级**，
  中文输出简体，且与 OmniVoice Studio 用同一个引擎 —— 两个项目维护成本减半。
* 本地权重在 `checkpoints/sensevoice-small/`（从 OmniVoice 项目直接复制即可，
  无需下载）。`use_itn=True` 才有标点，关掉会输出整段无标点（OmniVoice 实测）。

原来那段「whisper 转中文可能出繁体」的界面提示已随引擎一起删除 ——
SenseVoice 直接输出简体，不再需要向用户解释这个现象。
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

log = logging.getLogger("luxtts.asr")

_LOCK = threading.Lock()
_CACHE: dict = {}


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
    log.info("转写引擎 sensevoice：%s（纯 CPU，%d 线程）", d, n)
    return _SenseVoiceCaller(rec)


# --------------------------------------------------------------------------
# 可用性
# --------------------------------------------------------------------------
def unavailable_reason(engine: str | None = None) -> str | None:
    """返回「为什么转写用不了」，能用则返回 None。

    三种原因要分清楚，否则排障时会白找：整体关闭 / 缺 python 包 / 缺模型文件。
    """
    engine = (engine or config.ASR_ENGINE).strip().lower()
    if not config.ASR_ENABLED:
        return "自动转写已整体关闭（环境变量 LUXTTS_ASR=0）→ 新建音色时需要手工填写参考文本"
    if engine != "sensevoice":
        return f"未知的转写引擎：{engine!r}（本版本只支持 sensevoice）"
    import importlib.util

    if importlib.util.find_spec("sherpa_onnx") is None:
        return "缺少 python 包 sherpa-onnx（装：runtime\\python.exe -m pip install sherpa-onnx）"
    d = config.SENSEVOICE_DIR
    missing = [f for f in ("model.int8.onnx", "tokens.txt") if not (d / f).exists()]
    if missing:
        return f"缺少 SenseVoice 模型文件：{d}\\{', '.join(missing)}"
    return None


# --------------------------------------------------------------------------
# 单例
# --------------------------------------------------------------------------
def get_transcriber(device=None, force_reload: bool = False):
    """惰性构建转写器（进程内单例）。

    ``device`` 参数保留只是为了与旧调用点兼容 —— SenseVoice 走 sherpa-onnx
    的 int8 CPU 路径，**不用 GPU**，传什么都不影响。
    """
    with _LOCK:
        engine = "sensevoice"
        if engine in _CACHE and not force_reload:
            return _CACHE[engine]
        reason = unavailable_reason(engine)
        if reason is not None:
            log.warning("转写引擎不可用：%s", reason)
            return None
        obj = _build_sensevoice()
        _CACHE[engine] = obj
        return obj


def is_loaded(engine: str | None = None) -> bool:
    return "sensevoice" in _CACHE


def unload() -> None:
    with _LOCK:
        _CACHE.clear()
