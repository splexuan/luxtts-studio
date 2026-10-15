"""端到端冒烟：装模型 → 编码参考音频 → 生成中英各一段 → 落盘。

用法::

    runtime\\python.exe app\\smoke_test.py
    runtime\\python.exe app\\smoke_test.py --ref D:\\some\\voice.wav
    runtime\\python.exe app\\smoke_test.py --no-asr        # 强制走手工参考文本

不带 ``--ref`` 时，取 ``voices/`` 下的第一个 wav（音色库里随便哪个都行）。
落盘目录：``outputs/_smoke/``。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, core, hardware  # noqa: E402

ZH = "你好，这是 LuxTTS Studio 的冒烟测试，用来确认重构之后一切正常。"
EN = "Hello, this is the smoke test for LuxTTS Studio after the refactor."


def pick_ref(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"参考音频不存在：{p}")
        return p
    wavs = sorted(config.VOICES_DIR.glob("*.wav")) if config.VOICES_DIR.is_dir() else []
    if not wavs:
        raise SystemExit(
            f"voices/ 下没有 wav，请用 --ref 指定一个参考音频（项目：{config.VOICES_DIR}）"
        )
    return wavs[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="参考音频路径（wav/mp3 都行）")
    ap.add_argument("--no-asr", action="store_true", help="跳过自动转写，改用内置参考文本")
    ap.add_argument("--skip-smooth", action="store_true", help="不额外测一遍 return_smooth")
    args = ap.parse_args()

    out_dir = config.OUTPUTS_DIR / "_smoke"
    ref = pick_ref(args.ref)

    print("=" * 62)
    prof = hardware.active_profile()
    print(f"档位      : {prof['label']}")
    print(f"torch     : {prof['torch']} (cu{prof['cuda_built']}) | 线程 {prof['threads']}")
    print(f"参考音频  : {ref}")
    print("=" * 62, flush=True)

    t = time.perf_counter()
    core.load_model()
    print(f"[1] 模型加载         {time.perf_counter() - t:6.2f} s", flush=True)
    print("    device_info    :", core.device_info(), flush=True)

    # ---- 编码参考音频：把「冷启动 / 热态 / 给了文本」三种情况分开测 ----
    # 不分清楚会得出错误结论：第一次 encode 里含着 SenseVoice（sherpa-onnx）
    # 的首次初始化，把它算进「转写耗时」会高估好几倍。
    if args.no_asr:
        print("[2] 按 --no-asr 跳过自动转写", flush=True)
        prompt = core.encode_ref(ref, ref_text="手工填写的参考文本")
    else:
        t = time.perf_counter()
        prompt = core.encode_ref(ref)
        cold = time.perf_counter() - t

        t = time.perf_counter()
        core.encode_ref(ref)
        warm = time.perf_counter() - t

        t = time.perf_counter()
        core.encode_ref(ref, ref_text=prompt["ref_text"])
        given = time.perf_counter() - t

        print(f"[2] encode_ref 冷启动（含 SenseVoice 首次初始化）{cold:7.3f} s", flush=True)
        print(f"    encode_ref 热态（这才是转写本身的耗时）{warm:7.3f} s", flush=True)
        print(f"    给了 ref_text 再编码                   {given:7.3f} s"
              f"   ← 音色库缓存省掉的就是前两项", flush=True)

    print(f"    参考文本       : {prompt['ref_text'][:60]}{'…' if len(prompt['ref_text']) > 60 else ''}",
          flush=True)
    print(f"    参考时长       : {prompt['ref_seconds']} s", flush=True)
    print("    prompt 形状    :",
          {k: (tuple(v.shape) if hasattr(v, "shape") else f"list[{len(v)}]")
           for k, v in prompt.items() if k in core.PROMPT_KEYS}, flush=True)

    # ---- 生成 ----
    for tag, text in (("zh", ZH), ("en", EN)):
        t = time.perf_counter()
        out = core.synthesize(text, prompt)
        core.save_wav(out_dir / f"{tag}.wav", out["wav"], out["sr"])
        rt = out["seconds"] / out["elapsed"] if out["elapsed"] else 0
        print(
            f"[3] 生成 {tag}        {out['elapsed']:6.2f} s  →  "
            f"{out['seconds']:.2f}s @ {out['sr']} Hz  ≈{rt:.1f}× 实时  →  outputs/_smoke/{tag}.wav",
            flush=True,
        )

    # ---- return_smooth 会切到 24k 头，必须确认采样率真的变了 ----
    if not args.skip_smooth:
        out = core.synthesize(ZH, prompt, {"return_smooth": True})
        print(f"[4] return_smooth=True → sr={out['sr']} Hz（应为 24000，否则前端会变速）",
              flush=True)
        out2 = core.synthesize(ZH, prompt, {"return_smooth": False})
        print(f"    return_smooth=False → sr={out2['sr']} Hz（应为 48000）", flush=True)

    print("\nSMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
