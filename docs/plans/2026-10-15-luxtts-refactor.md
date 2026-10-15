# LuxTTS 重构实施方案（对齐 OmniVoice Studio 架构）

> **目标**：把 `D:\python_www\LuxTTS` 从「Gradio 单文件 demo」重构成与 OmniVoice Studio 同构的两层架构
> （`app/` 纯核心 + `web/` 自研界面），补上**音色库**与**合成记录**，并把散落在 `app.py`
> 与上游源码里的路径/下载问题一并修掉。
>
> **架构**：核心不再依赖 Gradio；界面走 FastAPI + 单个 HTML；参数与路径全部收敛到 `app/config.py`；
> `zipvoice/` 与 `linacodec/` 里的**上游代码不动**（只保留已打的 torchcodec 补丁）。
>
> **技术栈**：Python 3.12（复用现有 `python/` 便携运行时）、PyTorch 2.8.0+cu128 / onnxruntime、
> FastAPI + uvicorn、soundfile。**无需新增 heavy 依赖**（fastapi/uvicorn 已随 gradio 装好）。

---

## 0. 现状实测（数字都是本次跑出来的，不是推测）

`app.py` 共 146 行，**模型加载 + Gradio 界面写在同一文件里**；核心调用只有两步：

```python
lux_tts = LuxTTS("checkpoints", device=device, threads=2)   # 第 19 行
enc     = lux_tts.encode_prompt(audio_prompt, duration=ref_duration, rms=rms)
wav     = lux_tts.generate_speech(text, enc, num_steps, t_shift, speed, return_smooth)
```

headless 基线（GTX 1650 4G，参考音频用 OmniVoice 的 `voices/刘飞.wav`，`duration=5`）：

| 阶段 | 实测耗时 |
|---|---|
| `LuxTTS()` 模型加载 | **4.37 s** |
| `encode_prompt()`（含 Whisper 转写） | **4.51 s** |
| `generate_speech()` 中文，出 3.95 s 音频 | **0.97 s**（≈4× 实时） |
| `generate_speech()` 英文，出 4.63 s 音频 | **0.92 s**（≈5× 实时） |

输出采样率 **48 000 Hz**（`vocos.return_48k`）；`return_smooth=True` 时切到 24 kHz 档。

> ### ⚠️ 上面那个「4.51 s」的归因是错的，已被分段实测推翻
>
> 原表把 4.51 s 记成「`encode_prompt()` 本身慢」，并在 §B 据此认为 Whisper 是瓶颈。
> 重构后用 `tools/time-encode.py` 分段计时，**同机同音频**跑出来是这样：
>
> | 阶段 | 实测耗时 |
> |---|---|
> | `load_model()` | 6.16 s |
> | **ASR 权重装载（冷）** | **9.69 s** ← 真正的大头 |
> | **ASR 转写（权重已热）** | **3.38 s** |
> | **特征提取 + tokenize（`encode_ref`，带文本跳过 ASR）** | **0.67 s** |
> | 再跑一次 `encode_ref` | 0.01 s |
>
> 所以：4.51 s 那次里，**特征提取本身远不到 1 秒**；占大头的两个都是 ASR，
> 而且大头是**装载 Whisper 权重**（一次性），不是转写计算。
> 冷启动第一次建音色的真实总成本 ≈ **19.9 s**，其中 ASR 占 13 s（65%）。
>
> 这个修正**反而加强了「换 SenseVoice」的建议**，但理由变了：
> 不是「转写慢」，而是 **Whisper base 的 282 M 权重装载要 9.69 s**，
> 而 SenseVoice Small 只有 240 M 且是 onnx 冷启（OmniVoice 实测秒级）。
> 换掉之后首次建音色的等待能砍掉一大半。

`encode_prompt()` 返回的 dict（**这决定音色库能不能做**）：

| key | 类型/形状 | 可否落盘 |
|---|---|---|
| `prompt_tokens` | `list[list[int]]`，本例 56 个 | ✅ 纯 JSON |
| `prompt_features_lens` | `tensor int64 (1,)` | ✅ |
| `prompt_features` | `tensor float32 (1, 469, 100)` ≈ **188 KB** | ✅ |
| `prompt_rms` | `tensor float32` 标量 | ✅ |

→ **四个字段全可序列化，且体积很小** —— 音色 prompt 缓存完全可行（对齐 OmniVoice 的 `voices/<name>.pt`）。

### 0.1 顺带查出的问题（读代码 + 跑基线确认）

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| 1 | `load_models_cpu()` 第一行**无条件** `model_path = snapshot_download('YatharthS/LuxTTS')` | `zipvoice/modeling_utils.py:135` | **传入的 `checkpoints` 被无视**，CPU 模式每次联网重下 1.1 G |
| 2 | CPU 路径转写用 `openai/whisper-tiny`（联网）；GPU 路径用**硬编码相对路径** `whisper-base/snapshots/e37978b…` | 同上 `:106` `:142` | 本地已有的 `whisper-base/` 在 CPU 下不被使用；换工作目录就找不到模型 |
| 3 | `run.bat` 没有 `cd /d "%~dp0"`，`app.py` 又用相对路径 `"checkpoints"` | `run.bat:1-7` | 只有「双击」（CWD=脚本目录）能跑；从快捷方式/其他目录启动必挂 |
| 4 | `params.device = torch.device(device, 0)` 硬编码 GPU 0 | `modeling_utils.py:118` | 多卡机器无法指定 `cuda:1`；传入 `cpu` 会抛错（所幸 CPU 走另一分支） |
| 5 | Whisper 转出的参考文本是**繁体**：`在這個信息越來越碎片化的時代…` | 实测输出 | 参考文本直接进 `prompt_tokens`，繁体/简体的拼音归一化可能有细微差异（能跑通，但属质量隐患） |
| 6 | `k2` 未安装，Swoosh 回退 PyTorch 实现 | 启动警告 | 仅性能提示，非错误 |
| 7 | 项目**不是 git 仓库**，也无任何测试/验证脚本 | — | 重构没有回退点，也没有回归手段 |

---

## 1. 目标架构（与 OmniVoice Studio 一一对应）

```
LuxTTS/
├── python/                 便携 Python 3.12 + 依赖（9.0 G，不入库）   ← 对应 OmniVoice 的 runtime/
├── app/                    ★ 新增：核心层，不含任何界面
│   ├── core.py             模型加载 / encode_prompt / generate / 参数
│   ├── config.py           路径与默认参数（唯一真源）
│   ├── hardware.py         设备与精度自适应
│   ├── voices.py           音色库：落盘、指纹、缓存、增删改
│   ├── history.py          合成记录（outputs/history.json）
│   └── asr.py              参考音频转写（可插拔：Whisper / SenseVoice）
├── web/                    ★ 新增：界面层
│   ├── server.py           FastAPI 后端（只调 app/ 的模块级函数）
│   └── static/index.html   单文件前端（零外部依赖）
├── voices/                 ★ 新增：音色库（个人数据，不入库）
├── outputs/                ★ 新增：生成结果 + history.json（不入库）
├── zipvoice/               上游源码（不改）
├── linacodec/              上游源码（不改）
├── checkpoints/            模型权重（模型 + vocoder）
├── whisper-base/           本地 Whisper（转写用，若保留 Whisper 方案）
├── run_web.bat             ★ 新增：启动自研界面（端口 7867）
├── setup.bat               ★ 新增：环境引导（对齐 OmniVoice 的 bootstrap.py）
└── app_legacy_gradio.py    旧 app.py 改名保底，验证通过后再删
```

| 维度 | OmniVoice Studio | LuxTTS 目标 |
|---|---|---|
| 端口 | 7866 | **7867**（错开，可同时开） |
| 界面 | FastAPI + 单 HTML，四标签页 | 同构，**三标签页**（见 §2 决策 A） |
| 核心 | `app/app.py`，无界面代码 | `app/core.py`，无界面代码 |
| 音色 | `voices/<name>.pt` + json 指纹缓存 | `voices/<name>.pt` + json 指纹缓存（**结构不同：dict 而非 VoiceClonePrompt**） |
| 记录 | `outputs/history.json`，最多 300 条 | 同构 |
| 启动 | `run_web.bat` → `runtime\python.exe -s web\server.py` | `run_web.bat` → `python\python.exe -s web\server.py` |
| 端口来源 | `app/config.py` 的 `HOST/PORT`，server 读配置 | 同构（**别再硬编码**） |
| 验证 | `design-review/verify-web/*.js` 真跑界面 | 同构 |

---

## 2. 需要拍板的四个决策

### A. 标签页：三页，**不照搬四页**

OmniVoice 的「语音设计」（instruct：性别/年龄/口音/方言）在 ZipVoice 里**没有对应概念** ——
强行照搬会做出一个假功能。LuxTTS 真实的参数面是 6 个采样参数（`rms / ref_duration / t_shift /
num_steps / speed / return_smooth`），应当作为「合成设置」区放在合成页，而不是伪装成一页。

→ 建议：**语音合成 / 音色库 / 合成记录**。语种也不需要下拉：`EmiliaTokenizer` 内部
`get_segment()` 自动按中英分段（`tokenizer.py:257` 注释 "now only en and ch"）。

### B. 转写引擎：留 Whisper base，还是换 SenseVoice？

| | Whisper base（现状） | SenseVoice Small（对齐 OmniVoice） |
|---|---|---|
| 体积 | 282 M（已在本地） | 240 M（**OmniVoice 的 `checkpoints/sensevoice-small/` 可直接复制，无需下载**） |
| 依赖 | transformers（已装） | `sherpa-onnx`（**需新装**，LuxTTS 环境里没有） |
| 中文质量 | 实测输出**繁体** | 输出简体，中文更强 |
| 速度 | 慢，但**慢在权重装载（9.69 s）而非转写计算（3.38 s）**；特征提取本身只要 0.67 s | 快（OmniVoice 实测 0.5 s 量级，且 onnx 冷启远快于 transformers 装载） |
| 一致性 | 与 OmniVoice 两套引擎 | 两个项目**同一套**，维护成本减半 |

→ 建议 **换 SenseVoice**，理由如实测：中文更准（简体）+ 快 + 与 OmniVoice 统一。
**但请注意措辞上的区别**：OmniVoice 那边移除 Whisper 是因为它是 **1.6 G 的 large-v3-turbo**；
这里只是 **282 M 的 base**，收益没有那边大。所以这是一次「值得做但非必须」的优化 ——
**可以放到 Task 5 单独做，失败就退回 Whisper，不影响主重构。**

### C. 先建 git 基线（强烈建议）

LuxTTS **不是 git 仓库**、没有任何测试。重构前不建基线，改坏了无法回退 ——
而 OmniVoice 那边刚好踩过「删了 .git 又想推」的坑。建议 Task 0 先 `git init` + 首次提交
（`.gitignore` 补上 `python/`、`checkpoints/`、`voices/`、`outputs/`），本地提交即可，**不推远程**。

### D. 参考音频裁剪：**不需要做**

OmniVoice 的最大提速手段是「剪短参考音频」，因为它的前向耗时随参考帧数超线性增长。
LuxTTS 不同：`process_audio()` 里的 `librosa.load(..., duration=duration)` **已经**按
`ref_duration` 参数截断了（默认 5 s）。所以音色库**不必做波形裁剪器**，
只需保留「参考时长」这个参数即可。这是两个模型的结构差异，照搬会白做功。

---

## 3. 任务分解

> 约定：每个 Task 结束都在**真实运行**后确认（本项目不靠读代码下结论）。
> LuxTTS 无 pytest 测试套件，验证方式沿用 OmniVoice 的「跑真界面 + 看数字」。

### Task 0：建基线（不可跳过）

**Files**：Create `.gitignore`（在现有模板里追加）、`docs/plans/`（本文件所在处）

1. 在现有 `.gitignore`（GitHub Python 模板）末尾追加：
   `python/`、`checkpoints/`、`whisper-base/`、`voices/`、`outputs/`、`__pycache__/`
2. `git init` → `git add -A` → `git commit -m "chore: 重构前基线（Gradio 版，实测可用）"`
3. 确认 `git ls-files | wc -l` 数量合理（应只有源码与文档，**不含 9 G 的 `python/`**）

**验收**：`git status` 干净；`du -sh .git` < 5 MB。

### Task 1：抽出纯生成核心 `app/core.py`

**Files**：Create `app/__init__.py`、`app/core.py`；Modify 无（旧 `app.py` 原样保留）

要暴露的模块级函数（对齐 OmniVoice 的「少数符号即契约」原则）：

```python
load_model(force_reload=False)            # 单例，复用 LuxTTS 实例
is_loaded() -> bool
device_info() -> dict                     # 设备 / 模型名 / 显存
active_profile() -> dict                  # 默认采样参数
synthesize(text, audio, params) -> dict   # 返回 {"wav": np.ndarray, "sr": 48000, "seconds": float, "elapsed": float}
encode_ref(audio, duration, rms) -> dict  # 暴露给音色库
```

**关键约束（实测得出）**：
- `LuxTTS()` 的 `model_path` 必须是**绝对路径** `config.CHECKPOINTS`，不能再用 `"checkpoints"`（问题 #3）。
- 输出统一成 `np.float32` 单声道 + `sr=48000`；`return_smooth=True` 时采样率实际是 24 kHz，
  **必须把 `sr` 一起返回**，否则前端按 48 k 播放会变速。
- `generate_speech` 返回的是 CPU tensor，直接 `.numpy().squeeze()`。

**验收**：`python\python.exe -c "from app import core; core.load_model(); print(core.device_info())"`
→ 打印设备信息且不报错。

### Task 2：修掉上游的路径与下载问题（问题 #1 #2 #4）

**Files**：Modify `zipvoice/modeling_utils.py`（**只改这几处**，其余保持上游原样）

1. `load_models_cpu()` 里：`if model_path is None: model_path = snapshot_download(...)`
   （**去掉无条件下载**，与 `load_models_gpu()` 的写法对齐）
2. `transcriber` 的模型路径改为**由调用方传入**，默认解析成绝对路径；
   本地存在 `whisper-base/snapshots/<rev>/` 时**直接用本地**，不再触发下载
3. `torch.device(device, 0)` → 支持 `cuda:1` 这类写法（`torch.device("cuda:1")` 直接用；`cpu` 不加索引）

> 这三处是**上游代码的 bug**，但收益明确（省 1.1 G 重下 + 让 CPU 模式真能用）。
> 改动处都加 `# [LuxTTS-local]` 注释前缀，方便日后与上游 diff。

**验收**：临时改 `config.DEVICE="cpu"`，断网（或 `HF_HUB_OFFLINE=1`）跑一次，
确认**不触发下载**且能出声。

### Task 3：`app/config.py` + `app/hardware.py`

**Files**：Create `app/config.py`、`app/hardware.py`

- `config.py`：`ROOT / CHECKPOINTS / VOICES_DIR / OUTPUTS_DIR / WHISPER_DIR / HOST="127.0.0.1" /
  PORT=7867 / DEVICE="auto" / ASR_ENABLED / ASR_ENGINE` + 默认采样参数
  （`RMS=0.01 / REF_DURATION=5 / T_SHIFT=0.9 / NUM_STEPS=4 / SPEED=0.8 / RETURN_SMOOTH=False`）
  **路径一律用 `Path(__file__).resolve().parent.parent` 推导，杜绝相对路径**。
- `hardware.py`：`cuda > mps > cpu`；`active_profile()` 按设备给默认值
  （CPU 时自动降 `threads`、提示参考时长可调小）。

**验收**：从**任意目录**启动都能正确找到模型：
`cd C:\ && D:\python_www\LuxTTS\python\python.exe -c "from app import config; print(config.CHECKPOINTS.exists())"` → `True`

### Task 4：音色库 `app/voices.py`（本方案最大的新增价值）

**Files**：Create `app/voices.py`；Create `voices/`（运行时自建）

- 保存：`save_voice(name, audio_path, ref_text, duration, rms)` → 落盘 `voices/<name>.wav`
  + `voices/<name>.json`（`name / ref_text / duration / rms / created / duration_sec`）
- **prompt 缓存**：`voices/<name>.pt` = `torch.save(encode_dict)`，
  指纹存 json 的 `prompt_cache = {audio_size, audio_mtime, duration, rms, model_rev, asr_engine}`；
  命中则 `torch.load` 直接复用（对齐 OmniVoice 的 `_read_cached_prompt`）
- 列表/重命名/删除/改参考时长后重建 prompt
- `process_audio()` 每次都要重跑转写（**权重装载 9.69 s + 转写 3.38 s**）——**缓存后第二次生成应为 0.01 s**，
  这是最直观的收益

**验收**：同一音色连续生成两次，第二次 `encode_ref` 耗时 **< 0.1 s**
（对照冷启动首次 ≈19.9 s，其中特征提取仅 0.67 s）。
→ 实测 `tools/time-encode.py`：第二次 0.01 s，**达标且有50 倍余量**。

### Task 5：转写引擎可插拔 `app/asr.py`（决策 B 落地）

**Files**：Create `app/asr.py`

**重要发现：不用改上游代码就能换引擎** —— `LuxTTS` 把 `transcriber` 存成了**实例属性**
（`luxvoice.py:34`），而 `process_audio()` 接收的正是 `self.transcriber`。所以：

```python
lux = LuxTTS(...)
lux.transcriber = _sensevoice_caller   # 一个可调用对象即可
```

适配器只需满足 `transcriber({"array": np.ndarray, "sampling_rate": int}) -> {"text": str}`。

- 装 `sherpa-onnx`：`uv pip install --python python\python.exe sherpa-onnx`
- 模型：**直接从 `D:\python_www\omnivoice-studio\checkpoints\sensevoice-small\` 复制**（240 M，无需下载）
- `use_itn=True` 才有标点（OmniVoice 实测结论，直接沿用）
- 保留 Whisper 作为 `ASR_ENGINE="whisper"` 的退路；**换引擎前先在中文音频上对比转写文本**

**验收**：同一段中文参考音频，两个引擎各转一次，把文本并排打出来对比（重点看简/繁与错字）后再定。

### Task 6：FastAPI 后端 `web/server.py`

**Files**：Create `web/server.py`

接口（刻意与 OmniVoice 同构，便于两项目间迁移经验）：

```
GET  /api/bootstrap                     设备/参数默认值/音色数
GET  /api/voices                        音色列表（含 duration）
POST /api/voices                        保存新音色（上传音频 + 转写 + 建 prompt）
POST /api/voices/rename | /retrim | /retranscribe
POST /api/voices/delete
POST /api/transcribe                    试转写（不落盘）
POST /api/jobs                          提交合成任务 + 轮询进度
GET  /api/history | POST /api/history/delete | /api/history/clear
GET  /media/*                           产物读取（路径白名单 + 穿越校验）
```

**注意**：模型加载 4.37 s，**不要**在启动时就加载（对齐 OmniVoice 的懒加载，
这样界面验证脚本可以低成本反复跑）；首次合成再加载，`/api/bootstrap` 只报「未加载」。

### Task 7：单文件前端 `web/static/index.html`

**Files**：Create `web/static/index.html`

- 直接复用 OmniVoice 的**设计令牌体系与波形播放器**（`buildPlayer`：canvas 波形 +
  播放头 + 时间戳 + 点/拖跳转），**但采样率按 48 kHz 传**（OmniVoice 是别的采样率）
- 三标签页：语音合成 / 音色库 / 合成记录
- 合成页：选音色 → 填文本 → 6 个采样参数（带「默认」复位）→ 生成 + 进度轮询
- 音色库：上传 → 试转写（可手改）→ 保存 → 列表管理（**不做波形裁剪器**，见决策 D）
- 记录页：四行排版（身份/参数/正文/操作），统计项定宽 + `tabular-nums`；
  **`.hrow-top .badge{margin-left:0}`** 这条 OmniVoice 踩过的坑直接带上
- 一条 OmniVoice 的经验直接复制：**记录页默认不挂播放器**，点「试听」才 `buildPlayer`

### Task 8：启动与环境脚本

**Files**：Create `run_web.bat`、`setup.bat`；Modify 旧 `run.bat`（保留或改名）

**bat 三条硬规矩（OmniVoice 实测，直接沿用）**：**GBK(cp936) + CRLF + 不写 `chcp 65001`**。
现有 `run.bat` 就是 `chcp 65001` + 需要核对行尾 —— 属于要改的对象。

`run_web.bat` 内容对齐 OmniVoice：

```bat
@echo off
cd /d "%~dp0"                      rem ← 现有 run.bat 缺这行（问题 #3）
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_ENDPOINT=https://hf-mirror.com"
"%~dp0python\python.exe" -s "%~dp0web\server.py"
```

### Task 9：界面验证脚本 `design-review/verify-web/`

**Files**：Create `design-review/verify-web/run-verify.py`、`verify-voices.js`、`verify-history-layout.js`

照搬 OmniVoice 的跑法：`python\python.exe design-review\verify-web\run-verify.py <脚本> [端口]`
（自动「起服务 → 等 `/api/bootstrap` → 跑 node → 关服务」）。
两个已知坑直接沿用：**别用 `nohup ... &` 起服务**；**本机没有 playwright 自带浏览器**，
必须 `chromium.launch({channel:'chrome'})`。

### Task 10：文档

`README.md` 改为本项目说明（现为上游英文 README，写的是 `pip install -r requirements.txt` 的通用流程，
与本机「便携 runtime + 免安装」的实际用法不符）；新增 `操作教程.md`。

---

## 4. 工作量与风险

| 任务 | 量级 | 风险 |
|---|---|---|
| 0 基线 | 小 | 无 |
| 1 核心抽出 | 中 | 低（逻辑就是现有 `app.py` 的 `infer()`） |
| 2 修上游 bug | 小 | 中（动了 `zipvoice/`，需注释标记 + 回归一次 GPU/CPU 两条路） |
| 3 配置/硬件 | 小 | 低 |
| 4 音色库 | **大** | 中（`encode_dict` 序列化要实测往返一致；见下） |
| 5 SenseVoice | 中 | 中（新装依赖；**失败可退回 Whisper**） |
| 6 后端 | 中 | 低（接口可按 OmniVoice 抄） |
| 7 前端 | **大** | 中（播放器 48 kHz；音色库页面最费工） |
| 8/9/10 脚本与文档 | 小 | 低 |

**唯一真正的新技术风险**：`encode_dict` 的序列化往返。
`prompt_tokens` 是 `list`、其余是 tensor，`torch.save` 后 `torch.load` 应完全一致 ——
但**必须先写一个 10 行的往返断言实测通过再往下做**（不通过就别做缓存，退回每次重算）。

**第二风险**：Task 2 改动上游文件。若日后 `git pull` 上游，会冲突 —— 四行改动 + 注释标记可控。

---

## 5. 建议的执行顺序

```
Task 0（基线） → 1（核心） → 3（配置） → 6（后端骨架） → 7（前端，先只做合成页）
   → 4（音色库） → 9（验证脚本） → 2（修上游） → 5（SenseVoice，可选） → 8/10（收尾）
```

先用「核心 + 后端 + 合成页」打通一条最小可用链路（此时右侧已有成果可看），
再补音色库与记录页。**Task 2 与 Task 5 都排在主链路之后** —— 它们是优化，
不该阻塞「能跑起来」。

---

## 6. 可选增强（不在本次范围，供后续判断）

- **跨项目共享音色**：OmniVoice 的 `voices/*.wav` 可直接当 LuxTTS 的参考音频（本次基线就是这么测的）。
  两边的 prompt 格式不同（`VoiceClonePrompt` vs dict），但**参考音频层可以共用**。
- **ONNX/CPU 路径**：`checkpoints/` 里已有 `fm_decoder_int8.onnx`(124 M) + `text_encoder_int8.onnx`(5.5 M)，
  LuxTTS 天然有 CPU 降级路径（OmniVoice 只能退回 PyTorch CPU）。修完 Task 2 后这条路径才真正可用。
- **安装 `k2`**：消除启动时 "Failed import k2 … fallback to PyTorch" 的警告，Swoosh 走原生实现。
- **float16 推理**：官方 roadmap 未发布，README 称「应该快近 2×」，等上游。
