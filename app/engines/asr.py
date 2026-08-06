"""ASR 引擎: faster-whisper large-v3-turbo (CTranslate2, 本地离线)

统一接口: transcribe(audio_path) -> [Segment(start, end, text, words)]
"""
from __future__ import annotations

import glob
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()   # 保证 .env 中 VP_MODELS 生效 (与 audio.py 同款)

# 默认相对项目根定位 (K:\视频翻译与配音\models), 可被 VP_MODELS 覆盖
MODELS_ROOT = os.environ.get("VP_MODELS", str(Path(__file__).resolve().parents[3] / "models"))

# 本地无模型时自动下载的远端仓库 (CTranslate2 格式; 可被 VP_WHISPER_MODEL 覆盖)
DEFAULT_WHISPER_REPO = "Systran/faster-whisper-large-v3-turbo"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list | None = None

    def __repr__(self):
        return f"[{self.start:7.1f} - {self.end:7.1f}] {self.text}"


def resolve_whisper_dir() -> str:
    """定位本地 CTranslate2 模型目录 (兼容 local-dir / cache 两种布局);
    本地无模型时返回 HF repo id (WhisperModel 首次运行自动下载, 便于迁移部署).
    VP_WHISPER_MODEL 可强制指定 repo id 或本地路径"""
    explicit = os.environ.get("VP_WHISPER_MODEL")
    if explicit:
        return explicit
    root = os.path.join(MODELS_ROOT, "faster-whisper")
    for pattern in (
        os.path.join(root, "*"),                     # 平铺布局 (fetch_models 下载目标)
        os.path.join(root, "snapshots", "*"),        # HF 缓存布局
        os.path.join(root, "models--*", "snapshots", "*"),
    ):
        snaps = sorted(glob.glob(pattern))
        for s in snaps:
            if os.path.isfile(os.path.join(s, "model.bin")):
                return s
    print(f"[ASR] 未找到本地 faster-whisper 模型, 将自动从 HuggingFace 下载 "
          f"{DEFAULT_WHISPER_REPO} (~1.6GB, 仅首次运行; 国内可设 HF_ENDPOINT=https://hf-mirror.com)")
    return DEFAULT_WHISPER_REPO


class ASREngine:
    def __init__(self, model_dir: str | None = None, device: str | None = None,
                 compute_type: str = "float16", initial_prompt: str | None = None,
                 hotwords: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # 计算精度: VP_WHISPER_COMPUTE 显式覆盖 > cuda=float16 / cpu=int8 默认
        cfg_compute = os.environ.get("VP_WHISPER_COMPUTE")
        self.compute_type = cfg_compute or (compute_type if device == "cuda" else "int8")
        self.model_dir = model_dir or resolve_whisper_dir()
        # 新闻专有名词词表 (VP_ASR_PROMPT 注入 initial_prompt 提升识别)
        self.initial_prompt = initial_prompt if initial_prompt is not None \
            else os.environ.get("VP_ASR_PROMPT")
        # 热词: faster-whisper 强制候选词 (VP_ASR_HOTWORDS, 逗号分隔), 比 prompt 更硬性
        self.hotwords = hotwords if hotwords is not None else os.environ.get("VP_ASR_HOTWORDS")
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            print(f"[ASR] 加载模型: {self.model_dir} ({self.compute_type})")
            self.model = WhisperModel(self.model_dir, device=self.device,
                                      compute_type=self.compute_type)

    def transcribe(self, audio_path: str, language: str | None = None,
                   word_timestamps: bool = True,
                   vad_filter: bool = True,
                   beam_size: int = 5,
                   initial_prompt: str | None = None,
                   hotwords: str | None = None) -> list[Segment]:
        """转写音频 -> 带时间戳的分段列表
        initial_prompt/hotwords 传 None 时用构造时配置 (VP_ASR_PROMPT / VP_ASR_HOTWORDS)"""
        self._ensure_model()
        prompt = initial_prompt if initial_prompt is not None else self.initial_prompt
        hot = hotwords if hotwords is not None else self.hotwords
        segments, info = self.model.transcribe(
            audio_path, language=language, beam_size=beam_size,
            vad_filter=vad_filter, word_timestamps=word_timestamps,
            initial_prompt=prompt, hotwords=hot,
        )
        result = []
        for seg in segments:
            words = None
            if word_timestamps and seg.words:
                words = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
            result.append(Segment(seg.start, seg.end, seg.text.strip(), words))
        return result

    def release(self):
        """用完即释放显存"""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
