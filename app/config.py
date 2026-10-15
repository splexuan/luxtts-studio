"""路径与运行参数的**唯一真源**。

两条硬规矩，改代码时别绕开：

1. **所有路径都由本文件推导成绝对路径。** 上游那版是 `LuxTTS("checkpoints")`
   （相对路径）+ `run.bat` 里没有 `cd /d "%~dp0"`，只有在「双击 bat、CWD 恰好等于
   项目目录」时才对 —— 从快捷方式、计划任务、或任何别的目录启动都会找不到模型。
   所以这里一律用 `Path(__file__).resolve().parent.parent` 推导，任何模块都不许再写相对路径。

2. **界面端口只有这一处。** `web/server.py` 读 `HOST`/`PORT`，不许硬编码 ——
   否则配置形同虚设。

环境变量可以临时覆盖（都在下面各行的 `os.environ.get` 里）：
    LUXTTS_HOST / LUXTTS_PORT / LUXTTS_DEVICE / LUXTTS_THREADS
    LUXTTS_ASR=0（关掉自动转写）/ LUXTTS_ASR_ENGINE=sensevoice
    LUXTTS_ASR_THREADS / LUXTTS_SENSEVOICE_DIR
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 目录（全部绝对路径）
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

UPSTREAM = ROOT / "upstream"           # 上游源码：zipvoice/ linacodec/（保持原样，见 core.py 文件头）
CHECKPOINTS = ROOT / "checkpoints"     # 模型 + 声码器
VOCODER_DIR = CHECKPOINTS / "vocoder"
WHISPER_DIR = ROOT / "whisper-base"    # 本地 Whisper（转写退路用，不联网）
VOICES_DIR = ROOT / "voices"           # 音色库（个人数据）
OUTPUTS_DIR = ROOT / "outputs"         # 生成结果 + history.json
HISTORY_FILE = OUTPUTS_DIR / "history.json"

# --------------------------------------------------------------------------
# 服务
# --------------------------------------------------------------------------
HOST = os.environ.get("LUXTTS_HOST", "127.0.0.1")
PORT = _env_int("LUXTTS_PORT", 7867)   # 与 OmniVoice Studio 的 7866 错开，可同时开

# --------------------------------------------------------------------------
# 运行档位
# --------------------------------------------------------------------------
DEVICE = os.environ.get("LUXTTS_DEVICE", "auto")      # auto | cuda | cuda:1 | cpu | mps
NUM_THREADS = _env_int("LUXTTS_THREADS", 0)           # 0 = 自动（只在 CPU/ONNX 路径用到）

# --------------------------------------------------------------------------
# 参考音频转写
# --------------------------------------------------------------------------
ASR_ENABLED = _env_bool("LUXTTS_ASR", True)
ASR_ENGINE = os.environ.get("LUXTTS_ASR_ENGINE", "whisper").strip().lower()  # whisper | sensevoice
ASR_THREADS = _env_int("LUXTTS_ASR_THREADS", 0)       # 0 = 自动
SENSEVOICE_DIR = Path(
    os.environ.get("LUXTTS_SENSEVOICE_DIR") or (CHECKPOINTS / "sensevoice-small")
)

# --------------------------------------------------------------------------
# 生成参数
# --------------------------------------------------------------------------
# 这一组是上游 README / app.py 的推荐值，**所有设备都用同一份** ——
# 不给不同设备偷偷换默认值，免得「同一个音色在两台机器上结果不同」还找不到原因。
# 设备相关的建议只体现在界面的提示文案里（见 hardware.active_profile()）。
DEFAULTS: dict = {
    "rms": 0.01,            # 参考音频响度归一化目标；调高更响
    "ref_duration": 5.0,    # 只取参考音频的前 N 秒（越小越快，官方建议 ≥3 秒）
    "t_shift": 0.9,         # 采样参数；越高音质越好但更易读错字
    "num_steps": 4,         # 蒸馏模型 3~4 步即可，官方称 4 最优
    "speed": 0.8,           # 语速（内部会乘 1.3）
    "return_smooth": False, # True 会用 24k 的平滑头，音质更柔但不如 48k 清晰
}
PARAM_RANGES: dict = {
    "rms": (0.001, 0.1),
    "ref_duration": (1.0, 30.0),
    "t_shift": (0.1, 1.0),
    "num_steps": (1, 16),
    "speed": (0.5, 2.0),
}

# 参考音频的时长建议（秒）。**这是界面裁剪提示的唯一真源** ——
# 前端不再自己写一份数字，否则改了一处就会和这里对不上。
# 下限 2 秒是「还能听出是谁」的底线，3~10 秒是官方给的推荐区间。
TRIM_MIN_SECONDS = 2
TRIM_RECOMMENDED = (3, 10)

# 输出采样率由 return_smooth 决定，声码器有 48k / 24k 两个头。
# **返回时必须把 sr 一起带上**，否则前端按 48k 播 24k 的音频会变速。
SR_48K = 48000
SR_24K = 24000

# 合成记录上限（只记参数，不存音频副本）
HISTORY_MAX = 300
