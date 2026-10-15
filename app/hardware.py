"""设备解析与运行档位。

上游把设备名一路透传，并且 `params.device = torch.device(device, 0)` —— **硬编码 0 号卡**：
多卡机器没法用 `cuda:1`，而 `torch.device("cpu", 0)` 本身是非法的（CPU 路径恰好走了另一个
分支才没炸）。这里统一成「解析一次、全项目共用」，并顺手把显存读出来给界面显示。
"""

from __future__ import annotations

import os

import torch

from . import config

_CACHE: dict = {}


def resolve_device() -> torch.device:
    """把 ``config.DEVICE`` 解析成一个可用的 ``torch.device``。

    ``auto`` 的优先级：cuda > mps > cpu。显式写了就照写的来（写错了让 torch 自己报错，
    不吞掉 —— 静默降级比报错更难排查）。
    """
    raw = (config.DEVICE or "auto").strip().lower()
    if raw and raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def num_threads() -> int:
    """CPU / ONNX 路径的线程数。``config.NUM_THREADS=0`` 表示自动。"""
    if config.NUM_THREADS > 0:
        return config.NUM_THREADS
    return max(2, min(4, (os.cpu_count() or 4) // 2))


def gpu_info() -> dict | None:
    """当前卡的名字/显存/算力；没有可用 CUDA 时返回 None。"""
    if not torch.cuda.is_available():
        return None
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        return {
            "index": idx,
            "name": props.name,
            "total_gib": round(total / 2**30, 2),
            "free_gib": round(free / 2**30, 2),
            "capability": f"{props.major}.{props.minor}",
        }
    except Exception:  # noqa: BLE001 —— 读不到就算了，不能因为显示信息把主流程搞挂
        return None


def device_label(dev: torch.device | None = None) -> str:
    """给人看的一行设备描述。"""
    dev = dev if dev is not None else resolve_device()
    if dev.type == "cuda":
        info = gpu_info()
        if info:
            return f"{info['name']} · {info['total_gib']:.1f} GiB · cuda:{dev.index or 0}"
        return f"cuda:{dev.index or 0}"
    return dev.type


def active_profile() -> dict:
    """本机档位。界面用它渲染顶部信息条，也用来生成「注意事项」文案。

    **不**在这里偷偷改 ``config.DEFAULTS`` —— 不同设备拿到不同的默认参数会变成
    「同一音色两台机器结果不同」且极难排查。设备相关的建议只以 ``notes`` 形式给出。
    """
    if "profile" in _CACHE:
        return _CACHE["profile"]

    dev = resolve_device()
    threads = num_threads()
    notes: list[str] = []

    if dev.type == "cpu":
        notes.append(
            "当前跑在 CPU 上（ONNX 路径）。想更快可把「参考时长」降到 3~4 秒、"
            "「采样步数」降到 3。"
        )
    else:
        info = gpu_info()
        if info and info["total_gib"] < 6:
            notes.append(
                f"显存只有 {info['total_gib']:.1f} GiB，属入门卡。"
                "长文本建议一次少合成一点，或把「采样步数」降到 3。"
            )
    if not config.ASR_ENABLED:
        notes.append("自动转写已关闭（LUXTTS_ASR=0）：每次新建音色都要手工填参考文本。")

    lo, hi = config.TRIM_RECOMMENDED
    notes.append(f"参考音频建议 {lo}~{hi} 秒；低于 {config.TRIM_MIN_SECONDS} 秒克隆会不稳定。")

    profile = {
        "device": dev.type,
        "device_str": str(dev),
        "device_index": dev.index,
        "label": device_label(dev),
        "gpu": gpu_info(),
        "threads": threads,
        "torch": torch.__version__,
        "cuda_built": torch.version.cuda,
        "defaults": dict(config.DEFAULTS),
        "param_ranges": dict(config.PARAM_RANGES),
        "sr": {"clean": config.SR_48K, "smooth": config.SR_24K},
        # 裁剪提示交给界面渲染，别让前端再抄一份数字
        "trim": {"min": config.TRIM_MIN_SECONDS, "recommended": [lo, hi]},
        "notes": notes,
    }
    _CACHE["profile"] = profile
    return profile


def reset_cache() -> None:
    """改了 config 之后让档位重新计算（测试用）。"""
    _CACHE.clear()
