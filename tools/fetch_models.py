#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测并下载缺失的模型/工具 (faster-whisper / Qwen3-TTS / ffmpeg)

MDX 三模型官方无直链, 无法自动下载: 请从发布页(Release 附件/网盘)下载后
放入 models\\MDX_Net_Models\\  (MDX23C-8KFFT-InstVoc_HQ.ckpt / UVR-MDX-NET-Inst_HQ_3.onnx / Kim_Vocal_2.onnx)

用法:
    python tools/fetch_models.py          # 检测缺失并逐个下载 (需联网; 大文件较慢)
    python tools/fetch_models.py --check  # 只检测缺失, 不下载
"""
from __future__ import annotations

import os
import sys
import zipfile
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODELS = os.path.join(os.path.dirname(PROJECT_ROOT), "models")   # K:\视频翻译与配音\models
HUGGINGFACE_TTS = os.path.join(
    os.path.dirname(os.path.dirname(PROJECT_ROOT)),
    "HuggingFace", "models", "Qwen3-TTS-12Hz-1.7B-Base")

WHISPER_REPO = "Systran/faster-whisper-large-v3-turbo"
QWEN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
HF_ENDPOINT = "https://hf-mirror.com"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

MDX_FILES = ["MDX23C-8KFFT-InstVoc_HQ.ckpt",
             "UVR-MDX-NET-Inst_HQ_3.onnx", "Kim_Vocal_2.onnx"]


def _env_vp_models() -> str:
    """VP_MODELS 优先级: 进程环境 > .env > 默认(项目上级 models)"""
    if os.environ.get("VP_MODELS"):
        return os.environ["VP_MODELS"]
    try:
        with open(os.path.join(PROJECT_ROOT, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("VP_MODELS="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return _DEFAULT_MODELS


MODELS = _env_vp_models()
FFMPEG_BIN = os.path.join(MODELS, "ffmpeg", "bin")
MDX_DIR = os.path.join(MODELS, "MDX_Net_Models")
WHISPER_ROOT = os.path.join(MODELS, "faster-whisper")


# ---------------- 检测 ----------------
def _whisper_ok() -> bool:
    """与 asr.resolve_whisper_dir 一致的查找: 平铺 / snapshots / HF cache 布局均可"""
    import glob
    if not os.path.isdir(WHISPER_ROOT):
        return False
    for pattern in (os.path.join(WHISPER_ROOT, "*"),
                    os.path.join(WHISPER_ROOT, "snapshots", "*"),
                    os.path.join(WHISPER_ROOT, "models--*", "snapshots", "*")):
        for s in glob.glob(pattern):
            if os.path.isfile(os.path.join(s, "model.bin")):
                return True
    return False


def detect() -> dict[str, str]:
    """返回 {名称: 状态}, 状态: ok / missing / hint(mdx 需手动)"""
    out: dict[str, str] = {}
    out["faster-whisper"] = "ok" if _whisper_ok() else "missing"
    out["Qwen3-TTS"] = ("ok" if (os.path.isdir(HUGGINGFACE_TTS)
                                 and os.listdir(HUGGINGFACE_TTS))
                        else "missing")
    out["ffmpeg"] = ("ok" if os.path.isfile(os.path.join(FFMPEG_BIN, "ffmpeg.exe"))
                     else "missing")
    mdx = all(os.path.isfile(os.path.join(MDX_DIR, f)) for f in MDX_FILES)
    out["MDX 三模型"] = "ok" if mdx else "hint"
    return out


def _print_status(st: dict[str, str]) -> None:
    print("=" * 52)
    print("  模型/工具检查 (VP_MODELS = %s)" % MODELS)
    print("=" * 52)
    for name, s in st.items():
        mark = {"ok": "[OK]     ", "missing": "[MISSING] ", "hint": "[MANUAL]  "}[s]
        print(f"  {mark}{name}")


# ---------------- 下载 ----------------
def _download_hf(repo: str, local_dir: str, label: str) -> bool:
    print(f"  下载 {label} ({repo}, 经 {HF_ENDPOINT}) ... 文件较大请耐心等待")
    try:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=local_dir)
        return True
    except Exception as e:
        print(f"  [ERROR] 下载 {label} 失败: {e}")
        print("          请检查网络后重试, 或手动从 hf-mirror.com 下载并解压到:")
        print(f"          {local_dir}")
        return False


def _download_ffmpeg() -> bool:
    try:
        os.makedirs(FFMPEG_BIN, exist_ok=True)
        zip_path = os.path.join(MODELS, "ffmpeg", "_tmp_ffmpeg.zip")
        print(f"  下载 ffmpeg (gyan.dev, ~80MB) ...")
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if base in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
                    with z.open(name) as src, \
                         open(os.path.join(FFMPEG_BIN, base), "wb") as dst:
                        dst.write(src.read())
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"  [ERROR] 下载 ffmpeg 失败: {e}")
        print("          请从 https://www.gyan.dev/ffmpeg/builds/ 手动下载")
        print(f"          解压后把 bin 下的 exe 放入: {FFMPEG_BIN}")
        return False


def main() -> int:
    check_only = "--check" in sys.argv
    st = detect()
    _print_status(st)
    if check_only:
        print("  [CHECK] 以上为缺失检测结果 (未下载)")
        return 0

    missing = [k for k, v in st.items() if v == "missing"]
    if st["MDX 三模型"] == "hint":
        print()
        print("  ⚠️  MDX 三模型缺失: 官方无直链, 无法自动下载")
        print(f"      请从发布页 (Release 附件/网盘) 下载后放入:")
        print(f"      {MDX_DIR}")
        print(f"      需要: {', '.join(MDX_FILES)}")
    if not missing:
        print("  [OK] 可自动下载的模型均已就绪")
        return 0
    print()
    print(f"  将下载 {len(missing)} 项缺失资源 (大文件, 视网速可能需数分钟到数小时) ...")
    for name in missing:
        ok = False
        if name == "faster-whisper":
            ok = _download_hf(WHISPER_REPO, os.path.join(WHISPER_ROOT, "large-v3-turbo"),
                              "faster-whisper")
        elif name == "Qwen3-TTS":
            ok = _download_hf(QWEN_REPO, HUGGINGFACE_TTS, "Qwen3-TTS")
        elif name == "ffmpeg":
            ok = _download_ffmpeg()
        if not ok:
            return 1
        print(f"  [DONE] {name} 下载完成")
    print()
    print("  ✅ 全部缺失资源已处理。运行环境检查 ([2]) 确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
