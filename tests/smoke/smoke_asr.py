"""M1 冒烟测试: faster-whisper + large-v3-turbo (CTranslate2, 本地模型离线加载)

用法:
    python tests/smoke/smoke_asr.py <测试音频路径>

模型定位:
    models/faster-whisper/ 可能有两种布局（都兼容）:
      1. local-dir 布局:  snapshots/<hash>/model.bin   (hf download --local-dir)
      2. cache 布局:      models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/<hash>/model.bin
    直接传含 model.bin 的目录给 WhisperModel, 完全离线, 不联网。
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODELS_ROOT = os.environ.get("VP_MODELS", r"K:\视频翻译与配音\models")


def resolve_whisper_dir() -> str:
    root = os.path.join(MODELS_ROOT, "faster-whisper")
    for pattern in (
        os.path.join(root, "snapshots", "*"),                          # local-dir 布局
        os.path.join(root, "models--*", "snapshots", "*"),             # cache 布局
    ):
        snaps = sorted(glob.glob(pattern))
        for s in snaps:
            if os.path.isfile(os.path.join(s, "model.bin")):
                return s
    raise FileNotFoundError(f"未找到 faster-whisper 模型 (model.bin) in {root}")


def main(audio_path: str) -> int:
    from faster_whisper import WhisperModel

    model_dir = resolve_whisper_dir()
    print(f"[OK] 使用本地模型: {model_dir}")

    model = WhisperModel(model_dir, device="cuda", compute_type="float16")
    print("[OK] faster-whisper 模型加载成功")

    segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
    print(f"检测语言: {info.language} (置信度 {info.language_probability:.2f})")
    for seg in segments:
        print(f"[{seg.start:7.1f} - {seg.end:7.1f}] {seg.text}")
    print("[PASS] ASR 冒烟完成")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="测试音频路径")
    args = ap.parse_args()
    sys.exit(main(args.audio))
