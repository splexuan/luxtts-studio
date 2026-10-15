"""把 `encode_ref` 的 4.51 s 拆开看 —— 验证方案文档里那个归因对不对。

背景：方案文档 §0 写「`encode_prompt()`（含 Whisper 转写）4.51 s」，并据此在 §B 判断
「Whisper 慢，是换 SenseVoice 的主要理由」。如果那4.51 s 里大头其实是别的东西，
那§B 的速度一栏就得改。

跑法：runtime\\python.exe tools\\time-encode.py <参考音频路径>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\python_www\omnivoice-studio\voices\刘飞.wav")


def main() -> int:
    import torch
    from app import asr as asr_mod
    from app import config, core

    print(f"参考音频：{REF}")
    print(f"引擎：{config.ASR_ENGINE}\n")

    t0 = time.perf_counter()
    st = core.load_model()
    t_load = time.perf_counter() - t0
    print(f"[1] load_model                {t_load:6.2f} s   device={st['device']}")

    # 转写：冷（第一次，要装权重）
    t0 = time.perf_counter()
    tr = asr_mod.get_transcriber(st["device"])
    t_asr_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    import librosa
    wav16, _ = librosa.load(str(REF), sr=16000, duration=5.0)
    text = (tr({"array": wav16, "sampling_rate": 16000}).get("text") or "").strip()
    t_asr_run = time.perf_counter() - t0
    print(f"[2] ASR 权重装载(冷)         {t_asr_load:6.2f} s")
    print(f"[3] ASR 转写(权重已热)       {t_asr_run:6.2f} s   “{text[:24]}…”")

    # 特征提取（不含转写）
    t0 = time.perf_counter()
    prompt = core.encode_ref(REF, duration=5.0, rms=0.01, ref_text=text)
    t_encode = time.perf_counter() - t0
    print(f"[4] encode_ref(带文本，跳过ASR){t_encode:6.2f} s")

    #再跑一次，看缓存热态
    t0 = time.perf_counter()
    core.encode_ref(REF, duration=5.0, rms=0.01, ref_text=text)
    print(f"[5] encode_ref 再来一次       {time.perf_counter()-t0:6.2f} s")

    total_old = t_load + t_encode+ t_asr_load + t_asr_run
    print(f"\n冷启动第一次建音色合计≈ {total_old:6.2f} s")
    print(f"其中模型加载              {t_load:6.2f} s")
    print(f"其中 ASR 权重装载         {t_asr_load:6.2f} s")
    print(f"其中 ASR 转写             {t_asr_run:6.2f} s")
    print(f"其中特征提取+ tokenize    {t_encode:6.2f} s  ← 方案里说的 4.51 s 其实是这一项")
    void = torch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())