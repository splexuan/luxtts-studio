"""LuxTTS Studio —— 生成核心层。

本包**不含任何界面代码**（界面在 `web/`）。`web/server.py` 只调用本包的模块级函数；
命令行脚本（`app/smoke_test.py` 等）也走同一套接口。

模块分工：
    config.py    路径与运行参数的唯一真源（路径全部是绝对路径）
    hardware.py  设备解析与运行档位
    asr.py       参考音频转写（SenseVoice Small，唯一引擎）
    core.py      模型装配、参考音频编码、语音生成
    voices.py    音色库（含 prompt 缓存）
    history.py   合成记录
"""
