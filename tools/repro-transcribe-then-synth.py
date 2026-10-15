"""复现「转写之后紧接着合成」这条路径 —— verify-flow 卡住的那一步。

为什么单独写：浏览器那套变量太多（puppeteer / 页面 JS / 截图），
出问题时分不清是前端还是后端。这里只用 HTTP 直连，后端崩了就是后端崩。

跑法：
  起服务：runtime\\python.exe web\\server.py
  本脚本：runtime\\python.exe tools\\repro-transcribe-then-synth.py <wav>
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

BASE = f"http://{config.HOST}:{config.PORT}"
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\python_www\omnivoice-studio\voices\刘飞.wav")


def post_json(path: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json_bytes(payload),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(path: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def json_bytes(d: dict) -> bytes:
    return json.dumps(d).encode("utf-8")


def post_multipart(path: str, field: str, filename: str, blob: bytes,
                    extra: dict | None = None, timeout: float = 600.0) -> dict:
    """手搓 multipart。**不要用 requests** —— runtime 里没装，而且
    「用一个包少一个依赖」在便携运行时里是硬要求。"""
    boundary = "----LuxTTS" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in (extra or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\""
                     f"\r\n\r\n{v}\r\n".encode("utf-8"))
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8"))
    parts.append(blob)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    print(f"服务：{BASE}")
    print(f"参考音频：{REF}\n")
    blob = REF.read_bytes()

    print("[1] POST /api/transcribe（冷：要装 Whisper 权重，最容易崩的一步）")
    t0 = time.perf_counter()
    try:
        tr = post_multipart("/api/transcribe", "file", REF.name, blob)
        print(f"    {time.perf_counter()-t0:6.2f} s -> {tr['text'][:32]}…")
    except (urllib.error.URLError, OSError) as e:
        print(f"    FAILED: {e}")
        return 1

    print("\n[2] POST /api/voices（建音色 + prompt 缓存）—— multipart，与真实前端一致")
    t0 = time.perf_counter()
    name = f"复现用音色{time.strftime('%H%M%S')}"
    try:
        r = post_multipart("/api/voices", "file", REF.name, blob, extra={
            "name": name, "ref_text": tr["text"],
            "ref_duration": "5.0", "rms": "0.01",
        }, timeout=180)
        print(f"    {time.perf_counter()-t0:6.2f} s -> {r}")
    except (urllib.error.URLError, OSError, urllib.error.HTTPError) as e:
        print(f"    FAILED: {e}")
        return 1

    print("\n[3] POST /api/jobs（真跑生成 —— verify-flow 卡的就是这步）")
    t0 = time.perf_counter()
    try:
        job = post_json("/api/jobs", {
            "text": "这是一次复现验证，数字一二三四五。", "voice": name,
        })
        jid = job["job"]["id"]
        print(f"    提交成功 id={jid}")
    except (urllib.error.URLError, OSError) as e:
        print(f"    FAILED 提交阶段: {e}")
        print("    >>> 服务在这一步就没了。去看 _srv.log 末尾。")
        return 1

    print("[4] 轮询任务状态")
    deadline = time.time() + 420
    while time.time() < deadline:
        time.sleep(1)
        try:
            st = get_json(f"/api/jobs/{jid}")
        except (urllib.error.URLError, OSError) as e:
            print(f"    [1s] 服务已无响应: {e}")
            print("    >>> 服务在生成过程中退出。看 _srv.log 末尾。")
            return 1
        j = st
        print(f"    {j['state']:8s} {j['progress']:3d}%  {j['stage']}")
        if j["state"] in ("done", "error"):
            print(f"    耗时 {time.perf_counter()-t0:.2f} s")
            print(f"    {j}")
            return 0 if j["state"] == "done" else 1
    print("    超时")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())