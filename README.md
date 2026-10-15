# LuxTTS Studio

基于 [LuxTTS](https://github.com/ysharma3501/LuxTTS)（ZipVoice 蒸馏版）的本地语音克隆工作台。
单文件Web 界面，双击 `run_web.bat` 即用，无需打开浏览器折腾参数。

- **48 kHz 清晰输出**（不是常见的 24 kHz）
- **4 步出字**，GTX 1650上实测实时倍率约 **3.7×**
- **参考音频 3~10 秒**即可建音色，建完缓存在本地
- 纯本地推理，**不联网**（转写用随包Whisper，不调任何云 API）

---

## 快速开始

双击 **`run_web.bat`**，浏览器会自动打开 <http://127.0.0.1:7867/>。

模型**不在启动时加载**（加载要 5 秒，加载完界面才能秒开），第一次真正合成时才装载，
所以刚打开页面时不会有任何模型相关的等待。

### 三个页面

| 页面 | 干什么 |
| --- | --- |
| **语音合成** | 选音色 → 填文本 → 生成。生成完得到结果卡（波形播放器 + 4 项统计 + 本次参数） |
| **音色库** | 新建音色：上传参考音频 → 波形裁剪 → 试转写 → 保存。也管理已有音色的改名、重转写、删除 |
| **合成记录** | 历史列表，可试听 / 重新生成 / 删除 |

### 新建一个音色

1. 进「音色库」→ 点「新建音色」
2. 上传一段 3~10 秒的录音（**波形上直接拖拽裁剪**，落盘的永远是当时听到的那一段）
3. 点「试转写」自动填参考文本 —— 也可以自己改
4. 命名并保存。**保存即建好prompt 缓存**，之后合成直接命中，不用再编码

> 参考文本很重要：它会和目标文本一起进模型，改了参考文本就必须重建 prompt。
> 入口在音色库详情的「重新转写参考文本」。

---

## 生成参数

界面上的四个参数都是**合成期**参数。想改参考音频的响度（rms）和时长（ref_duration），
要去**音色库**的详情面板 —— 那两个在建prompt 时就已经固化了，放在合成页是假控件。

| 参数 | 默认 | 范围 | 说明 |
| --- | --- | --- | --- |
| `num_steps` | 4 | 1–16 | 蒸馏模型步数。**3~4 就够**，调高只是变慢 |
| `speed` | 0.8 | 0.5–2.0 | 语速。内部会乘 1.3 |
| `t_shift` | 0.9 | 0.1–1.0 | 越高音质越好，但更容易读错字 |
| `return_smooth` | 关 | — | 开=24 kHz 更柔和，关=48 kHz 更清晰 |

默认值来自上游 README，**所有设备共用同一份** —— 不按设备偷换默认值，
免得「同一个音色在两台机器上结果不同」还找不到原因。

输出采样率由 `return_smooth` 决定：关 → 48000 Hz，开 → 24000 Hz。

---

## 换一台机器

仓库里**只有源代码**。下面这些是外部资产，体积合计约 11 GB，需要手动摆进去：

```
<项目目录>/
├─ runtime/        便携Python 3.12 + 全部依赖   ~9.0 GB
├─ checkpoints/    模型权重 + 声码器            ~1.1 GB
│  ├─ config.json
│  ├─ model.pt  tokens.txt  text_encoder.onnx  fm_decoder.onnx
│  └─ vocoder/  vocos.bin  config.yaml
└─ whisper-base/   本地 Whisper（自动转写用）    ~282 MB
```

摆好之后双击 **`setup.bat`** 逐项自检（它只检查、不会下载 11 GB 的东西——
那种脚本比告诉你要装什么更糟），然后 `run_web.bat` 启动。

| 缺什么 | 能做什么 |
| --- | --- |
| `runtime/` | 完全跑不起来 |
| `checkpoints/` | 完全跑不起来 |
| `upstream/` | 完全跑不起来（**这个在仓库里**，2.3 M） |
| `whisper-base/` | 能跑，建音色时必须手工填参考文本 |

###环境变量

都需要**先设好再启动**（默认走 `app/config.py` 里的值）：

```
LUXTTS_HOST / LUXTTS_PORT          监听地址，默认 127.0.0.1:7867
LUXTTS_DEVICE                      auto | cuda | cuda:1 | cpu
LUXTTS_THREADS                     CPU/ONNX 路径的线程数，0 = 自动
LUXTTS_ASR=0                       关掉自动转写
LUXTTS_ASR_ENGINE                  whisper | sensevoice
```

---

## 代码结构

```
app/                生成核心，**不含任何界面代码**
├─ config.py        路径与参数的唯一真源（端口也在这）
├─ hardware.py      探测 GPU/CPU 档位，给界面提示文案
├─ core.py          模型装配 + 参考编码 + 采样（复用上游函数）
├─ voices.py        音色库：prompt 缓存的落盘与读取
├─ asr.py           参考音频转写
└─ history.py       合成记录（只记参数，不存音频副本）
web/
├─ server.py        FastAPI 后端：收 HTTP → 调 app/ → 存盘 → 回 JSON
└─ static/index.html单文件前端，零外部依赖，自持浅/深设计令牌
upstream/           上游 LuxTTS 源码快照（**保持零diff**）
```

### 三条硬规矩

1. **端口只有一处来源**：`app/config.py` 的 `HOST`/`PORT`，`web/server.py` 读它，不许硬编码。
   浏览器由 `web/server.py --open` 打开，不要在 bat 里用 `for /f` 读 Python 输出来拿端口 ——
   命令串以双引号开头时 cmd 根本不执行，内层单引号又会提前截断，两种失败都**静默无声**。
2. **`.bat` 必须是 GBK + CRLF + 纯 ASCII，且不写 `chcp 65001`**。四条缺一不可：
   非 CRLF 会闪退无报错；UTF-8 + chcp 会把中文行读成乱码。写盘用
   `p.write_bytes(text.replace("\n","\r\n").encode("gbk"))`。
3. **音频只用 `soundfile`，不重采样**。`app/core.py` 是唯一的生成核心。

---

## 验证

```bash
# 界面冒烟（不加载模型，秒级）
runtime\python.exe design-review\verify-web\run-verify.py verify-ui

# 端到端真跑：上传 → 转写 → 保存 → 合成 → 记录 → 删除
runtime\python.exe design-review\verify-web\run-verify.py verify-flow
```

两个坑：①别用 `nohup ... &` 起服务；②本机没有 playwright 自带浏览器，
脚本必须走 `chromium.launch({channel:"chrome"})` 用系统 Chrome。

---

## 已知限制

- **只支持 Windows**。便携运行时是 Windows 构建；CPU/ONNX 路径也只在 Windows 上验证过。
- **4 GB 显存是当前配置的下限**。`num_steps` 已按蒸馏模型调到 4；
  更大的步数在 4 GB 卡上会打满显存。
- **合成记录只存参数不存音频副本**，删了记录不会删 `outputs/` 里的 wav。
- 参考音频超过 10 秒会明显变慢（序列长度 = 参考帧 + 文本帧）。**在界面上裁短是最有效的提速手段。**