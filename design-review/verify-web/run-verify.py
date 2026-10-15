"""界面验证的编排器 —— 一条命令跑完「起服务 → 等就绪 → 跑 node 脚本 → 关服务」。

用法：
    runtime\\python.exe design-review\\verify-web\\run-verify.py verify-ui
    runtime\\python.exe design-review\\verify-web\\run-verify.py verify-flow [参考音频路径]

为什么要有这个脚本：起停服务、等端口、传参、收尾这几步每次都一样，
手工做容易漏（尤其是「服务没关干净」，下一次跑就撞端口）。

两个坑（都踩过）：

1. **别用 `nohup ... &` 起服务。** 那种方式下进程的输出管子容易在
   子进程还在跑的时候被回收掉，崩溃瞬间的现场就丢了。这里用
   `subprocess.Popen` 直接持有句柄。
2. **本机没有 playwright 自带浏览器。** verify 脚本里必须写
   `chromium.launch({ headless: true, channel: "chrome" })` 走系统 Chrome。

服务本身**不加载模型**（懒加载），所以跑 verify-ui 几乎不花钱；
verify-flow 会真跑一次生成，那一次要十几秒。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / "runtime" / "python.exe"
SHOTS = HERE / "shots"

sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_ready(host: str, port: int, proc: subprocess.Popen, limit: float = 60.0) -> bool:
    """等端口可连。**顺带看进程是否已经死了** —— 否则会白等满 limit 秒。"""
    deadline = time.time() + limit
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[FAIL] 服务提前退出，返回码 {proc.returncode}", flush=True)
            return False
        if port_open(host, port):
            return True
        time.sleep(0.3)
    return False


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    which = argv[1]
    script = HERE / f"{which}.js"
    if not script.exists():
        print(f"[FAIL] 找不到 {script}")
        return 2

    SHOTS.mkdir(parents=True, exist_ok=True)

    # node 侧依赖（puppeteer-core）装在托管的 workspace 里，靠 NODE_PATH 找。
    node_dir = Path.home() / ".workbuddy/binaries/node/workspace/node_modules"
    node = None
    for cand in sorted(Path.home().glob(".workbuddy/binaries/node/versions/*/node.exe"), reverse=True):
        node = cand
        break
    if node is None:
        print("[FAIL] 找不到 node.exe")
        return 2

    if port_open(config.HOST, config.PORT):
        print(f"[ERROR] {config.HOST}:{config.PORT} 已被占用 —— "
              f"多半是上一次的服务没关干净。先把它停掉。")
        return 1

    env = dict(os.environ)
    env["NODE_PATH"] = str(node_dir)
    env["PYTHONUTF8"] = "1"

    print(f"== 起服务 {config.HOST}:{config.PORT} ==", flush=True)
    # 服务输出**落盘而不是 PIPE**。PIPE 的缓冲在子进程被硬杀时（CUDA 崩溃就是
    # 这种死法）会跟着一起丢，事后什么都看不到；写文件至少能留下崩溃前的最后几行。
    srv_log = ROOT / "_srv.log"
    log_fh = srv_log.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [str(PYTHON), "-u", str(ROOT / "web" / "server.py")],
        cwd=str(ROOT), env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_ready(config.HOST, config.PORT, proc):
            _dump_tail(srv_log)
            return 1
        print("[ok] 服务就绪\n", flush=True)

        cmd = [str(node), str(script), str(SHOTS), str(config.PORT)]
        if which == "verify-flow":
            # verify-flow 要真上传一段参考音频。默认取第一段可用素材。
            ref = Path(argv[2]) if len(argv) > 2 else _default_ref()
            if ref is None:
                print("[FAIL] verify-flow 需要一段参考音频：\n"
                      "      runtime\\python.exe design-review\\verify-web\\run-verify.py "
                      "verify-flow <音频路径>")
                return 2
            print(f"[ok] 参考音频：{ref}\n", flush=True)
            cmd.append(str(ref))

        print(f"== 跑 {which}.js ==", flush=True)
        rc = subprocess.call(cmd, env=env, cwd=str(HERE))

        # 脚本跑完了但服务已经没了 —— 这种必须说，否则人会以为是脚本自己的问题。
        if proc.poll() is not None:
            print(f"\n[ERROR] 服务在验证过程中退出了，返回码 {proc.returncode}。"
                  f"\n        崩溃现场见 {srv_log} 的末尾：")
            _dump_tail(srv_log)
        return rc
    finally:
        print("\n== 关服务 ==", flush=True)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log_fh.close()
        # 万一 terminate 没弄干净（Windows 上偶尔会），别把端口留给下一次。
        for _ in range(20):
            if not port_open(config.HOST, config.PORT):
                break
            time.sleep(0.25)
        else:
            print(f"[WARN] {config.PORT} 仍被占用，下次跑之前先清掉")


def _dump_tail(path: Path, lines: int = 30) -> None:
    """打印日志末尾。**按 utf-8 + replace 读** —— 中文 Windows 的输出常含非 UTF-8
    字节，用默认编码读会抛 UnicodeDecodeError，然后异常被吞、什么都看不到。"""
    try:
        content = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    except OSError as e:
        print(f"        （读不到 {path}：{e}）")
        return
    for line in content[-lines:]:
        print("        " + line)


def _default_ref() -> Path | None:
    """找一个能当参考音频的素材：优先本项目的 voices/，否则翻上游的。"""
    for folder in (ROOT / "voices", ROOT.parent / "omnivoice-studio" / "voices"):
        if folder.is_dir():
            for wav in sorted(folder.glob("*.wav")):
                return wav
    return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))