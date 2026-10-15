"""合成记录。

存 ``outputs/history.json``（追加 + 原子替换，最多 ``config.HISTORY_MAX`` 条）。
**只记参数，不存音频副本** —— 音频是 ``outputs/`` 下的独立文件。

两条与 OmniVoice Studio 一致的约定：

1. 成功**和失败都要记**。界面上的错误提示一刷新就没了，回头排查只能靠记录。
2. 「删除记录 / 清空记录」**只动 json，绝不删 outputs 里的 wav** ——
   那是产物，要清得让用户自己清。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from . import config

log = logging.getLogger("luxtts.history")

_LOCK_FILE_SUFFIX = ".lock"


def _ensure_dir() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_all() -> list[dict]:
    p = config.HISTORY_FILE
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("合成记录损坏，按空处理：%s", e)
        return []
    return data if isinstance(data, list) else []


def _write_all(items: list[dict]) -> None:
    """原子替换：先写临时文件再 os.replace，避免中途失败留下半个 json。"""
    _ensure_dir()
    p = config.HISTORY_FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# --------------------------------------------------------------------------
# 产物命名
# --------------------------------------------------------------------------
_SAFE = re.compile(r"[^\w\u4e00-\u9fff-]+")


def next_output_path(text: str, voice: str | None = None) -> Path:
    """给本次合成起一个不重名的文件名：``<时间>_<音色>_<文本前几个字>.wav``。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    parts = [stamp]
    if voice:
        parts.append(_SAFE.sub("", voice)[:16] or "voice")
    snippet = _SAFE.sub("", (text or "").strip())[:12]
    if snippet:
        parts.append(snippet)
    else:
        parts.append(hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:6])
    path = config.OUTPUTS_DIR / ("_".join(parts) + ".wav")
    if path.exists():   # 同一秒内连点两次
        path = path.with_name(f"{path.stem}_{uuid.uuid4().hex[:4]}.wav")
    return path


# --------------------------------------------------------------------------
# 增删查
# --------------------------------------------------------------------------
def add(
    *,
    ok: bool,
    text: str,
    voice: str | None = None,
    params: dict | None = None,
    output: str | Path | None = None,
    sr: int | None = None,
    seconds: float | None = None,
    elapsed: float | None = None,
    error: str | None = None,
    ref_text: str | None = None,
    mode: str = "voice",
) -> dict:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": _now(),
        "ok": bool(ok),
        "mode": mode,                       # voice=用音色库；adhoc=临时参考音频
        "voice": voice,
        "text": text,
        "params": params or {},
        "output": str(output) if output else None,
        "sr": sr,
        "seconds": seconds,
        "elapsed": elapsed,
        "error": error,
    }
    if ref_text:
        entry["ref_text"] = ref_text[:200]
    items = _read_all()
    items.append(entry)
    if len(items) > config.HISTORY_MAX:
        items = items[-config.HISTORY_MAX:]
    _write_all(items)
    return entry


def list_items(limit: int = 100, offset: int = 0) -> dict:
    """倒序返回。每条带 ``exists`` 标记产物文件是否还在（不主动删记录）。"""
    items = _read_all()
    total = len(items)
    items = list(reversed(items))
    window = items[offset: offset + limit]
    for it in window:
        out = it.get("output")
        it["exists"] = bool(out) and Path(out).exists()
    return {"items": window, "total": total, "offset": offset, "limit": limit}


def delete(ids: list[str]) -> dict:
    """按 id 删除记录。**只动 json。**"""
    target = set(ids or [])
    items = _read_all()
    kept = [it for it in items if it.get("id") not in target]
    removed = len(items) - len(kept)
    if removed:
        _write_all(kept)
    return {"removed": removed, "kept": len(kept)}


def clear() -> dict:
    """清空记录。**只动 json，outputs/ 里的 wav 全部保留。**"""
    items = _read_all()
    _write_all([])
    return {"removed": len(items)}


def count() -> int:
    return len(_read_all())


def stats() -> dict:
    items = _read_all()
    ok = sum(1 for i in items if i.get("ok"))
    return {"total": len(items), "ok": ok, "failed": len(items) - ok, "max": config.HISTORY_MAX}
