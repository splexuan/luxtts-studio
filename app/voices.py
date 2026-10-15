"""音色库：参考音频 + 参考文本 + **prompt 缓存**。

上游每次都要重新上传参考音频、重新转写、重新编码（首次编码要十几秒，其中大部分是
Whisper 权重装载与 CUDA 预热）。这里把结果缓存下来：``voices/<名>.pt``。

目录布局（与 OmniVoice Studio 同构）::

    voices/
      <名>.wav    参考音频 —— **统一存成 24 kHz 单声道**
      <名>.json   元信息 + prompt 指纹
      <名>.pt     prompt 缓存（torch.save 的 dict）

为什么音频要归一成 24 kHz 单声道：整条链路只用得到 24 kHz（声学特征）和 16 kHz（转写），
而 16 kHz 由 24 kHz 重采样即可。存原始 44.1 kHz 立体声只是白占空间、并在每次编码时
多一次重采样。json 里的 ``stored_sr`` 记录了这件事。

缓存失效条件（任一变化就重建 prompt）见 :func:`prompt_fingerprint`。
**注意 ``ref_text`` 也算在内**：参考文本会进 ``prompt_tokens`` 参与生成，
只改 json 里的文字而不重建 prompt 是不生效的（OmniVoice 项目踩过同一个坑）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path

from . import config, core

log = logging.getLogger("luxtts.voices")

STORE_SR = config.SR_24K
_DUR_CACHE: dict[str, float] = {}

# 文件名安全：只允许中英文、数字、空格、下划线、点、横线、括号。
# 音色名直接当文件名用，**必须挡住路径穿越**（名字来自界面输入）。
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10)),
}


class VoiceError(Exception):
    """音色库的可预期错误（界面直接显示 message）。"""


# --------------------------------------------------------------------------
# 基础
# --------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """把界面传来的名字校验成一个安全的文件名（不含扩展名）。"""
    raw = unicodedata.normalize("NFC", (name or "").strip())
    if not raw:
        raise VoiceError("音色名不能为空")
    name_clean = _BAD_CHARS.sub("", raw).strip().strip(".")
    if not name_clean:
        raise VoiceError(f"音色名 {raw!r} 里没有可用字符")
    if name_clean.lower() in _RESERVED:
        raise VoiceError(f"音色名 {name_clean!r} 是系统保留名，换一个")
    if len(name_clean) > 64:
        raise VoiceError("音色名太长（最多 64 个字符）")
    return name_clean


def voice_path(name: str) -> Path:
    return config.VOICES_DIR / f"{safe_name(name)}.wav"


def meta_path(name: str) -> Path:
    return config.VOICES_DIR / f"{safe_name(name)}.json"


def prompt_path(name: str) -> Path:
    return config.VOICES_DIR / f"{safe_name(name)}.pt"


def _ensure_dir() -> None:
    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _unique_name(base: str) -> str:
    base = safe_name(base)
    if not voice_path(base).exists() and not meta_path(base).exists():
        return base
    for i in range(2, 1000):
        cand = f"{base} ({i})"
        if not voice_path(cand).exists() and not meta_path(cand).exists():
            return cand
    raise VoiceError(f"{base} 的重名太多了")


def read_meta(name: str) -> dict | None:
    p = meta_path(name)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("音色元信息损坏 %s: %s", p, e)
        return None
    # 参考音频路径容错：json 里可能存着绝对路径，项目一旦被整体搬走就失效。
    # 音频本来就和 json 同目录，指不到就回退到同名 wav（只读不写回）。
    ref = meta.get("ref_audio")
    if ref and not Path(ref).exists():
        meta["ref_audio"] = str(voice_path(name))
    elif not ref:
        meta["ref_audio"] = str(voice_path(name))
    return meta


def _write_meta(name: str, meta: dict) -> None:
    _atomic_write_text(meta_path(name), json.dumps(meta, ensure_ascii=False, indent=2))


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def audio_duration(name: str) -> float:
    """参考音频时长（秒）。带缓存 —— 列表页一屏几十条，别每条都去解一次音频。"""
    key = str(voice_path(name))
    if key in _DUR_CACHE:
        return _DUR_CACHE[key]
    p = voice_path(name)
    if not p.exists():
        return 0.0
    try:
        import soundfile as sf

        info = sf.info(str(p))
        dur = round(info.frames / info.samplerate, 2)
    except Exception:  # noqa: BLE001
        dur = 0.0
    _DUR_CACHE[key] = dur
    return dur


# --------------------------------------------------------------------------
# prompt 指纹
# --------------------------------------------------------------------------
def prompt_fingerprint(name: str, meta: dict) -> dict:
    """参与生成的所有输入。**任何一个变了，缓存就必须重建。**"""
    p = voice_path(name)
    st = p.stat() if p.exists() else None
    return {
        "audio_size": st.st_size if st else 0,
        "audio_mtime": int(st.st_mtime) if st else 0,
        "ref_duration": float(meta.get("ref_duration") or config.DEFAULTS["ref_duration"]),
        "rms": float(meta.get("rms") or config.DEFAULTS["rms"]),
        "ref_text": (meta.get("ref_text") or "").strip(),
        "asr_engine": config.ASR_ENGINE,
        "store_sr": STORE_SR,
    }


def _fp_hash(fp: dict) -> str:
    return hashlib.sha1(json.dumps(fp, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# prompt 读写
# --------------------------------------------------------------------------
def _save_prompt(name: str, prompt: dict) -> None:
    import torch

    _ensure_dir()
    torch.save(prompt, prompt_path(name))


def _load_prompt(name: str) -> dict | None:
    import torch

    p = prompt_path(name)
    if not p.exists():
        return None
    try:
        return torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 —— 缓存坏了就重建，不该让音色打不开
        log.warning("prompt 缓存读取失败 %s：%s（将重建）", p, e)
        return None


def read_prompt(name: str, *, rebuild: bool = False) -> dict:
    """取音色的 prompt。命中缓存直接返回（**毫秒级**），否则重建并落盘。"""
    meta = read_meta(name)
    if meta is None:
        raise VoiceError(f"音色不存在：{name}")
    audio = Path(meta["ref_audio"])
    if not audio.exists():
        raise VoiceError(f"音色的参考音频丢失：{audio}")

    fp = prompt_fingerprint(name, meta)
    if not rebuild and meta.get("prompt_cache", {}).get("hash") == _fp_hash(fp):
        cached = _load_prompt(name)
        if cached is not None:
            return cached
        log.info("音色 %s 的 prompt 缓存缺失，重建", name)

    prompt = core.encode_ref(
        audio,
        duration=fp["ref_duration"],
        rms=fp["rms"],
        ref_text=fp["ref_text"] or None,
    )
    _save_prompt(name, prompt)
    meta["prompt_cache"] = {"hash": _fp_hash(fp), "built_at": _now(), "fingerprint": fp}
    # 转写补全：新建时可能是手工填的，这里把转写结果回写进元信息
    meta["ref_text"] = prompt.get("ref_text", meta.get("ref_text"))
    meta["ref_seconds"] = prompt.get("ref_seconds", meta.get("ref_seconds"))
    _write_meta(name, meta)
    return prompt


def drop_prompt(name: str) -> None:
    try:
        prompt_path(name).unlink(missing_ok=True)
    except OSError as e:
        log.warning("删除 prompt 缓存失败 %s: %s", name, e)
    meta = read_meta(name)
    if meta is not None:
        meta.pop("prompt_cache", None)
        _write_meta(name, meta)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 增删改查
# --------------------------------------------------------------------------
def _store_audio(src: str | Path, dst: Path) -> float:
    """把源音频解码后存成 24 kHz 单声道 wav，返回时长。"""
    import librosa
    import numpy as np
    import soundfile as sf

    wav, _ = librosa.load(str(src), sr=STORE_SR, mono=True)
    if wav.size == 0:
        raise VoiceError("参考音频是空的，或格式无法解码")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), np.asarray(wav, dtype=np.float32), STORE_SR)
    return round(len(wav) / STORE_SR, 2)


def save_voice(
    name: str,
    src_audio: str | Path,
    *,
    ref_text: str | None = None,
    ref_duration: float | None = None,
    rms: float | None = None,
    auto_rename: bool = True,
) -> dict:
    """新建音色：落盘音频 + 元信息，**并同步建好 prompt 缓存**。

    点「保存」就等于音频与转写都已落盘（不留跨页补步骤）。转写与编码在这一步完成，
    所以保存会花掉一次编码的时间（冷启动十几秒，热态不到 1 秒）。
    """
    _ensure_dir()
    want = safe_name(name)
    final = _unique_name(want) if auto_rename else want
    if final != want and not auto_rename:
        raise VoiceError(f"音色已存在：{want}")

    src = Path(src_audio)
    if not src.exists():
        raise VoiceError(f"参考音频不存在：{src}")

    dst = voice_path(final)
    duration = _store_audio(src, dst)
    _DUR_CACHE.pop(str(dst), None)

    meta = {
        "name": final,
        "ref_audio": str(dst),
        "stored_sr": STORE_SR,
        "source_name": src.name,
        "ref_seconds": duration,
        "ref_duration": float(ref_duration or config.DEFAULTS["ref_duration"]),
        "rms": float(rms if rms is not None else config.DEFAULTS["rms"]),
        "ref_text": (ref_text or "").strip(),
        "created": _now(),
        "updated": _now(),
    }
    _write_meta(final, meta)

    # 建 prompt（会顺带把转写结果回写进 meta）
    try:
        read_prompt(final, rebuild=True)
    except VoiceError:
        raise
    except Exception as e:  # noqa: BLE001
        # 音频与元信息已经落盘了，不该因为建缓存失败就整个回滚 ——
        # 提示用户「音色已保存，但缓存没建起来」，下次生成会自动重试。
        log.warning("音色 %s 已保存，但 prompt 缓存建立失败：%s", final, e)
        meta["prompt_error"] = str(e)
        _write_meta(final, meta)

    return read_meta(final) or meta


def update_voice(
    name: str,
    *,
    ref_text: str | None = None,
    ref_duration: float | None = None,
    rms: float | None = None,
    rebuild: bool = True,
) -> dict:
    """改参考文本 / 参考时长 / 响度。改了就会重建 prompt（旧缓存作废）。"""
    meta = read_meta(name)
    if meta is None:
        raise VoiceError(f"音色不存在：{name}")
    changed = False
    if ref_text is not None and ref_text.strip() != (meta.get("ref_text") or "").strip():
        meta["ref_text"] = ref_text.strip()
        changed = True
    if ref_duration is not None and float(ref_duration) != float(meta.get("ref_duration") or 0):
        meta["ref_duration"] = float(ref_duration)
        changed = True
    if rms is not None and float(rms) != float(meta.get("rms") or 0):
        meta["rms"] = float(rms)
        changed = True
    if not changed:
        return meta
    meta["updated"] = _now()
    _write_meta(name, meta)
    if rebuild:
        read_prompt(name, rebuild=True)
    return read_meta(name) or meta


def retranscribe(name: str, *, duration: float | None = None) -> dict:
    """重新转写参考文本（用当前 ASR 引擎），并重建 prompt。"""
    if not config.ASR_ENABLED:
        raise VoiceError("自动转写已关闭（LUXTTS_ASR=0），无法重新转写")
    meta = read_meta(name)
    if meta is None:
        raise VoiceError(f"音色不存在：{name}")
    audio = Path(meta["ref_audio"])
    if not audio.exists():
        raise VoiceError(f"音色的参考音频丢失：{audio}")

    from . import asr as asr_mod

    transcriber = asr_mod.get_transcriber(core.load_model()["device"])
    if transcriber is None:
        raise VoiceError(f"转写引擎不可用：{asr_mod.unavailable_reason()}")

    import librosa

    dur = duration if duration is not None else meta.get("ref_duration")
    wav, _ = librosa.load(str(audio), sr=16000, duration=None if not dur or dur <= 0 else float(dur))
    text = (transcriber({"array": wav, "sampling_rate": 16000}).get("text") or "").strip()
    if not text:
        raise VoiceError("转写结果为空，请手工填写参考文本")
    return update_voice(name, ref_text=text)


def rename_voice(old: str, new: str) -> dict:
    old_meta = read_meta(old)
    if old_meta is None:
        raise VoiceError(f"音色不存在：{old}")
    target = safe_name(new)
    if target == safe_name(old):
        return old_meta
    if voice_path(target).exists() or meta_path(target).exists():
        raise VoiceError(f"音色已存在：{target}")

    for src, dst in (
        (voice_path(old), voice_path(target)),
        (meta_path(old), meta_path(target)),
        (prompt_path(old), prompt_path(target)),
    ):
        if src.exists():
            os.replace(src, dst)

    meta = read_meta(target) or old_meta
    meta["name"] = target
    meta["ref_audio"] = str(voice_path(target))
    meta["updated"] = _now()
    _write_meta(target, meta)
    _DUR_CACHE.pop(str(voice_path(old)), None)
    return meta


def delete_voices(names: list[str]) -> dict:
    """删除音色。逐文件捕获失败 —— 一个删不掉不该让整批都失败。"""
    done, failed = [], []
    for name in names:
        try:
            clean = safe_name(name)
        except VoiceError as e:
            failed.append({"name": name, "error": str(e)})
            continue
        if read_meta(clean) is None and not voice_path(clean).exists():
            failed.append({"name": clean, "error": "音色不存在"})
            continue
        errs = []
        for p in (voice_path(clean), meta_path(clean), prompt_path(clean)):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                errs.append(f"{p.name}: {e}")
        _DUR_CACHE.pop(str(voice_path(clean)), None)
        if errs:
            failed.append({"name": clean, "error": "；".join(errs)})
        else:
            done.append(clean)
    return {"deleted": done, "failed": failed}


def list_voices() -> list[dict]:
    """音色列表（按更新时间倒序）。带 ``duration``、``has_prompt`` 与是否「偏短」的提示。"""
    _ensure_dir()
    out: list[dict] = []
    for p in sorted(config.VOICES_DIR.glob("*.json")):
        meta = read_meta(p.stem)
        if meta is None:
            continue
        dur = audio_duration(meta["name"])
        meta["duration"] = dur
        meta["has_prompt"] = prompt_path(meta["name"]).exists()
        meta["audio_exists"] = voice_path(meta["name"]).exists()
        # 官方明确说参考音频低于 3 秒克隆会不稳定；超过 15 秒则纯属拖慢
        if dur and dur < 3:
            meta["length_hint"] = "偏短"
        elif dur > 15:
            meta["length_hint"] = "偏长"
        else:
            meta["length_hint"] = ""
        out.append(meta)
    out.sort(key=lambda m: (m.get("updated") or "", m.get("created") or ""), reverse=True)
    return out


def get_voice(name: str) -> dict:
    meta = read_meta(name)
    if meta is None:
        raise VoiceError(f"音色不存在：{name}")
    meta["duration"] = audio_duration(meta["name"])
    meta["has_prompt"] = prompt_path(meta["name"]).exists()
    meta["length_hint"] = (
        "偏短" if meta["duration"] and meta["duration"] < 3
        else ("偏长" if meta["duration"] > 15 else "")
    )
    return meta


def copy_from(src_audio: str | Path, dst: Path) -> Path:
    """把界面上的临时上传拷进项目（仅内部用）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_audio, dst)
    return dst
