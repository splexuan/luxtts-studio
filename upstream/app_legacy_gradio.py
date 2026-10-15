import sys
import subprocess
import time

import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 核心：代码内设置 HF 镜像 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import gradio as gr
import torch
from zipvoice.luxvoice import LuxTTS

# Init Model
device = "cuda" if torch.cuda.is_available() else "cpu"
lux_tts = LuxTTS("checkpoints", device=device, threads=2)


def infer(
    text,
    audio_prompt,
    rms,
    ref_duration,
    t_shift,
    num_steps,
    speed,
    return_smooth,
):
    if audio_prompt is None or not text:
        return None, "❗ 请同时输入文本和参考音频"

    start_time = time.time()

    # Encode reference
    encoded_prompt = lux_tts.encode_prompt(
        audio_prompt,
        duration=ref_duration,
        rms=rms,
    )

    # Generate speech
    final_wav = lux_tts.generate_speech(
        text,
        encoded_prompt,
        num_steps=int(num_steps),
        t_shift=t_shift,
        speed=speed,
        return_smooth=return_smooth,
    )

    duration = round(time.time() - start_time, 2)

    final_wav = final_wav.cpu().squeeze(0).numpy()
    final_wav = (np.clip(final_wav, -1.0, 1.0) * 32767).astype(np.int16)

    stats_msg = f"✨ 生成完成，用时 **{duration} 秒**"
    return (48000, final_wav), stats_msg


# =======================
# Gradio UI
# =======================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ LuxTTS 语音克隆演示")

    gr.Markdown(
        """
        > **说明：** 当前示例默认运行在 **2 核 CPU** 上，推理速度可能较慢。  
        > **使用建议：**  
        > - 如果出现“吞字 / 断词”，请 **降低语速（Speed）** 或 **增大参考音频时长**  
        > - 参考音频越清晰，克隆效果越稳定
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="待合成文本",
                value="你好，这是一个语音克隆的示例效果。",
            )

            input_audio = gr.Audio(
                label="参考音频（WAV 格式）",
                type="filepath",
            )

            with gr.Row():
                rms_val = gr.Number(
                    value=0.01,
                    label="RMS 音量（响度）",
                )
                ref_duration_val = gr.Number(
                    value=5,
                    label="参考音频时长（秒）",
                    info="越小越快，若出现音质异常可设置较大值（如 10~20）",
                )
                t_shift_val = gr.Number(
                    value=0.9,
                    label="T-Shift（音色偏移）",
                )

            with gr.Row():
                steps_val = gr.Slider(
                    1,
                    10,
                    value=4,
                    step=1,
                    label="采样步数（Steps）",
                )
                speed_val = gr.Slider(
                    0.5,
                    2.0,
                    value=0.8,
                    step=0.1,
                    label="语速（越小越慢 / 越清晰）",
                )
                smooth_val = gr.Checkbox(
                    label="启用平滑输出",
                    value=False,
                )

            btn = gr.Button("开始生成语音", variant="primary")

        with gr.Column():
            audio_out = gr.Audio(label="生成结果")
            status_text = gr.Markdown("🟢 等待生成中…")

    btn.click(
        fn=infer,
        inputs=[
            input_text,
            input_audio,
            rms_val,
            ref_duration_val,
            t_shift_val,
            steps_val,
            speed_val,
            smooth_val,
        ],
        outputs=[audio_out, status_text],
    )

demo.launch()
