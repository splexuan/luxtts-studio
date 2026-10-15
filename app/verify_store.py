"""自检：音色库、prompt 缓存、合成记录。

用法::

    runtime\\python.exe app\\verify_store.py
    runtime\\python.exe app\\verify_store.py --ref D:\\some\\voice.wav

**不会碰真实数据**：整个过程把 ``config.VOICES_DIR`` / ``config.HISTORY_FILE``
临时指向 ``outputs/_verify/``，跑完删掉。

重点验证 prompt 缓存的往返一致性 —— 这是「音色库缓存」能不能做的前提：
``encode_ref`` 返回的 dict 里既有 ``list[list[int]]`` 又有 tensor，
只要 ``torch.save`` / ``torch.load`` 之后有一个字节不同，缓存就是错的，
且错了很难看出来（合成出来的声音只是「稍微不对」）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, core, history, voices  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  —— {detail}" if detail else ""), flush=True)
    return ok


def pick_ref(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"参考音频不存在：{p}")
        return p
    wavs = sorted(config.VOICES_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"voices/ 下没有 wav，请用 --ref 指定参考音频：{config.VOICES_DIR}")
    return wavs[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="参考音频路径")
    ap.add_argument("--name", default="_自检音色", help="自检用的音色名（会被清理）")
    args = ap.parse_args()

    real_ref = pick_ref(args.ref)

    # ---- 把数据目录重定向到 outputs/_verify/，不碰真实音色库 ----
    sandbox = config.OUTPUTS_DIR / "_verify"
    if sandbox.exists():
        shutil.rmtree(sandbox, ignore_errors=True)
    (sandbox / "voices").mkdir(parents=True, exist_ok=True)
    config.VOICES_DIR = sandbox / "voices"
    config.HISTORY_FILE = sandbox / "history.json"
    core.load_model()          # 一次加载，后面共用
    src = sandbox / "src.wav"
    shutil.copyfile(real_ref, src)

    name = args.name
    print("=" * 62)
    print("1) prompt 序列化往返一致性（本轮最关键的一项）")
    print("=" * 62)
    prompt = core.encode_ref(src, ref_text="这是一段用于自检的参考文本。")
    import torch

    p_file = sandbox / "roundtrip.pt"
    torch.save(prompt, p_file)
    back = torch.load(str(p_file), map_location="cpu", weights_only=False)

    same_keys = set(back.keys()) == set(prompt.keys())
    check("key 集合一致", same_keys, f"{sorted(prompt.keys())}")

    detail = []
    all_equal = True
    for k in core.PROMPT_KEYS:
        a, b = prompt[k], back[k]
        if torch.is_tensor(a):
            # 比数值要搬到同一设备上比（缓存是 map_location="cpu" 读回来的，原张量在 cuda）
            a2, b2 = (a.cpu(), b.cpu()) if torch.is_tensor(b) else (a.cpu(), None)
            eq = (
                torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a2, b2)
            )
        else:
            eq = a == b
        all_equal &= bool(eq)
        detail.append(f"{k}={'同' if eq else '★异'}")
    check("四个 prompt 字段逐位一致（dtype/形状/数值）", all_equal, " ".join(detail))
    check("ref_text 一并保留", back.get("ref_text") == prompt["ref_text"])

    # 从磁盘读回来的缓存，张量在 CPU 上；合成时必须搬到模型设备，否则 forward 会报
    # 「Expected all tensors to be on the same device」。
    dev = core.load_model()["device"]

    def on_device(t) -> bool:
        # 注意：torch.device("cuda") != torch.device("cuda:0")，直接 == 比会假报失败。
        if not torch.is_tensor(t):
            return True
        return t.device.type == dev.type and (dev.index is None or t.device.index == dev.index)

    moved = core._prompt_only(back, dev)
    check(
        "缓存回读后张量已对齐到模型设备",
        all(on_device(t) for t in moved[1:]),
        f"目标设备 {dev}，实际 {[str(t.device) for t in moved[1:] if torch.is_tensor(t)]}",
    )
    out = core.synthesize("用磁盘上读回来的缓存合成一句话。", back)
    core.save_wav(sandbox / "from_cache.wav", out["wav"], out["sr"])
    check("直接用缓存合成成功", out["seconds"] > 0.2,
          f"{out['seconds']}s @ {out['sr']}Hz，{out['elapsed']}s")

    print()
    print("=" * 62)
    print("2) 音色库 增 / 查 / 改 / 重命名 / 删")
    print("=" * 62)
    meta = voices.save_voice(name, src, ref_text="自检用参考文本")
    check("保存音色", meta["name"] == name, f"name={meta['name']}")
    check("音频已落盘", voices.voice_path(name).exists())
    check("prompt 缓存已建立", voices.prompt_path(name).exists())
    check("参考时长已记录", voices.audio_duration(name) > 0, f"{voices.audio_duration(name)} s")

    fp1 = voices.prompt_fingerprint(name, voices.read_meta(name))
    check("首次读取命中缓存（不再重算）", voices.read_meta(name)["prompt_cache"]["hash"] == voices._fp_hash(fp1))

    voices.update_voice(name, ref_duration=3.0)
    fp2 = voices.prompt_fingerprint(name, voices.read_meta(name))
    check("改了 ref_duration → 指纹变化（缓存正确失效）", voices._fp_hash(fp1) != voices._fp_hash(fp2))

    voices.update_voice(name, ref_text="换一段完全不同的参考文本")
    fp3 = voices.prompt_fingerprint(name, voices.read_meta(name))
    check("改了 ref_text → 指纹变化（文本会进 prompt_tokens）",
          voices._fp_hash(fp2) != voices._fp_hash(fp3))

    lst = voices.list_voices()
    check("列表能列出来", any(v["name"] == name for v in lst), f"共 {len(lst)} 个")

    dup = voices.save_voice(name, src, ref_text="重名")
    check("重名自动改名", dup["name"] != name, f"→ {dup['name']}")

    new_name = f"{name}-改名后"
    voices.rename_voice(name, new_name)
    check("重命名后文件齐备",
          voices.voice_path(new_name).exists() and voices.meta_path(new_name).exists(),
          new_name)

    print()
    print("=" * 62)
    print("3) 安全：音色名里的路径穿越必须被挡住")
    print("=" * 62)
    for bad in ("../evil", "..\\evil", "a/b", "con", "  ", "x" * 80):
        try:
            got = voices.safe_name(bad)
            if got == bad or "/" in got or "\\" in got or ".." in got:
                check(f"拒绝 {bad!r}", False, f"竟然放行成 {got!r}")
            else:
                check(f"拒绝 {bad!r}", True, f"净化成 {got!r}")
        except voices.VoiceError as e:
            check(f"拒绝 {bad!r}", True, str(e))
    check("路径穿越未落盘",
          not (sandbox / "evil.wav").exists() and not (config.OUTPUTS_DIR / "evil.wav").exists())

    print()
    print("=" * 62)
    print("4) 合成记录")
    print("=" * 62)
    history.add(ok=True, text="记录测试", voice=new_name, params={"num_steps": 4},
                output="outputs/does-not-exist.wav", sr=48000, seconds=1.0, elapsed=0.5)
    history.add(ok=False, text="失败也要记", voice=new_name, error="示例错误")
    got = history.list_items(limit=10)
    check("成功与失败都记下了", got["total"] == 2, f"total={got['total']}")
    check("倒序（最新的在前）", got["items"][0]["text"] == "失败也要记")
    check("exists 标记（文件不存在时 False）", got["items"][1]["exists"] is False)

    history.delete([got["items"][0]["id"]])
    check("按 id 删除", history.count() == 1)
    history.clear()
    check("清空", history.count() == 0)

    print()
    print("=" * 62)
    print("5) 删除音色（只删三个文件，失败逐条报告）")
    print("=" * 62)
    res = voices.delete_voices([new_name, dup["name"], "不存在的音色"])
    check("两个真音色删掉", sorted(res["deleted"]) == sorted([new_name, dup["name"]]),
          f"deleted={res['deleted']}")
    check("不存在的音色进 failed", len(res["failed"]) == 1, f"failed={res['failed']}")

    print()
    print("=" * 62)
    ok = all(r[1] for r in results)
    print(f"共 {len(results)} 项，通过 {sum(1 for r in results if r[1])}，失败 {sum(1 for r in results if not r[1])}")
    print("STORE_OK" if ok else "STORE_FAILED")
    print("=" * 62)
    shutil.rmtree(sandbox, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
