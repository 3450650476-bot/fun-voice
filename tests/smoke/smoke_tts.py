"""M1 冒烟测试: Qwen3-TTS-12Hz-1.7B-Base 声音克隆合成中文

用法:
    python tests/smoke/smoke_tts.py --ref <参考音频> [--ref-text <转录文本>] [--text <合成文本>] [--out <输出wav>]

说明:
    - 参考音频: 3-30 秒干净人声 (可用 workspace/samples/ref_zh.mp3)
    - --ref-text: 参考音频的转录文本, 提供时克隆质量更佳; 缺省走 x-vector 模式
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

QWEN3_MODEL = os.environ.get("QWEN3_TTS_MODEL", r"K:\HuggingFace\models\Qwen3-TTS-12Hz-1.7B-Base")


def main(ref_audio: str, text: str, out_wav: str, ref_text: str = "") -> int:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        QWEN3_MODEL,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("[OK] Qwen3-TTS 模型加载成功")

    if ref_text:
        wavs, sr = model.generate_voice_clone(text=text, language="Chinese", ref_audio=ref_audio, ref_text=ref_text)
    else:
        prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text="", x_vector_only_mode=True)
        wavs, sr = model.generate_voice_clone(text=text, language="Chinese", voice_clone_prompt=prompt)

    sf.write(out_wav, wavs[0], sr)
    print(f"[PASS] 合成完成: {out_wav} ({len(wavs[0]) / sr:.2f}s @ {sr}Hz)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考音频路径 (3-30s 人声)")
    ap.add_argument("--ref-text", default="", help="参考音频转录文本(可选)")
    ap.add_argument("--text", default="你好，这是一段声音克隆测试。人工智能正在改变内容创作的方式。", help="要合成的文本")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "smoke_tts_out.wav"), help="输出 wav 路径")
    args = ap.parse_args()
    sys.exit(main(args.ref, args.text, args.out, args.ref_text))
