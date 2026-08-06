"""TTS 引擎: Qwen3-TTS-12Hz-1.7B-Base 零样本声音克隆

统一接口: clone_synthesize(text, ref_audio, ref_text) -> (wav, sr)
参考音色: 3-30s 干净人声; 提供 ref_text 时克隆相似度更高
"""
from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()   # 保证 .env 中 QWEN3_TTS_MODEL / VP_MODELS 生效 (与 audio.py 同款)


def _env_float(name: str):
    v = os.environ.get(name)
    return float(v) if v else None


def _env_int(name: str):
    v = os.environ.get(name)
    return int(v) if v else None

# 默认相对项目根: {VP_MODELS}/Qwen3-TTS-12Hz-1.7B-Base
# 实际位置 (如 K:\HuggingFace\models\...) 由 .env 的 QWEN3_TTS_MODEL 提供
_MODELS_ROOT = os.environ.get("VP_MODELS", str(Path(__file__).resolve().parents[3] / "models"))
QWEN3_MODEL = os.environ.get(
    "QWEN3_TTS_MODEL",
    str(Path(_MODELS_ROOT) / "Qwen3-TTS-12Hz-1.7B-Base"),
)


class TTSEngine:
    # 参考音色最长秒数 (超长截取; 可用 VP_REF_SECONDS 配置)
    MAX_REF_SECONDS = float(os.environ.get("VP_REF_SECONDS", 12))
    # 批量合成默认值: 一次 generate 的句数 (可用 VP_TTS_BATCH_SIZE env 或构造参数覆盖)
    # ⚠️ 两个坑: 单批过大(如 67 句)解码序列过长, 无 flash-attn 时极慢/卡死;
    #           过小(接近逐句)连续 generate 也会卡死 → 默认 12 是实测折中

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 batch_size: int | None = None,
                 temperature: float | None = None, top_p: float | None = None,
                 top_k: int | None = None, repetition_penalty: float | None = None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path or QWEN3_MODEL
        self.device = device
        self.model = None
        self._prompt = None          # 参考音频 prompt 缓存 (整段配音只计算一次)
        # 生成参数 (可选, 默认用 qwen_tts 默认; 可用 VP_TTS_* env 配置)
        self.batch_size = batch_size if batch_size is not None \
            else (_env_int("VP_TTS_BATCH_SIZE") or 12)
        self.temperature = temperature if temperature is not None else _env_float("VP_TTS_TEMPERATURE")
        self.top_p = top_p if top_p is not None else _env_float("VP_TTS_TOP_P")
        self.top_k = top_k if top_k is not None else _env_int("VP_TTS_TOP_K")
        self.repetition_penalty = repetition_penalty if repetition_penalty is not None \
            else _env_float("VP_TTS_REPETITION_PENALTY")

    def _ensure_model(self):
        if self.model is None:
            from qwen_tts import Qwen3TTSModel
            print(f"[TTS] 加载模型: {self.model_path}")
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=self.device,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

    @staticmethod
    def _clip_ref_audio(ref_audio: str, offset: float = 0.0) -> str:
        """超长参考音频截取 MAX_REF_SECONDS 秒 (Qwen3-TTS 官方 3s 即可克隆)
        offset: 从指定秒数开始截取 (原视频人声模式传 ASR 第一段起点, 避免截到片头音乐)"""
        import soundfile as sf
        import numpy as np
        try:
            info = sf.info(ref_audio)
            total = info.frames / info.samplerate
            if total <= TTSEngine.MAX_REF_SECONDS and offset <= 0:
                return ref_audio
            start = min(int(offset * info.samplerate),
                        max(0, int(total * info.samplerate) - 1))
            wave, sr = sf.read(ref_audio, dtype="float32", start=start,
                               frames=int(TTSEngine.MAX_REF_SECONDS * info.samplerate))
            # 临时文件放系统 temp (不污染项目包目录, 并发安全)
            clip = os.path.join(tempfile.gettempdir(), f"funvoice_ref_{os.getpid()}.wav")
            sf.write(clip, wave, sr)
            print(f"[TTS] 参考音频 {total:.0f}s 超长, 从 {offset:.0f}s 截取 {TTSEngine.MAX_REF_SECONDS}s")
            return clip
        except Exception:
            return ref_audio

    @staticmethod
    def _validate_ref_audio(ref_audio: str) -> None:
        """参考音色有效性校验: 无法读取/过短/近静音 → 中文报错 (防止克隆失败或音色漂移)"""
        import numpy as np
        import soundfile as sf
        try:
            info = sf.info(ref_audio)
        except Exception:
            raise ValueError(f"参考音色无法读取: {ref_audio} (文件损坏或格式不支持), 请更换音色")
        dur = float(info.frames) / float(info.samplerate)
        if dur < 0.5:
            raise ValueError(f"参考音色过短: 仅 {dur:.2f}s (<0.5s), 请更换更长的干净人声")
        frames = int(min(12.0, dur) * info.samplerate)
        data, _ = sf.read(ref_audio, dtype="float32", frames=frames)
        rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
        if rms < 1e-4:
            raise ValueError("参考音色近静音 (RMS 过低), 无法提取音色, 请更换干净人声")

    def build_prompt(self, ref_audio: str, ref_text: str | None = None,
                     ref_offset: float = 0.0):
        """预构建参考音频 prompt (跨多次生成复用, 提升整段配音一致性)
        ⚠️ 踩坑: qwen_tts 的 voice_clone_prompt 若含文本(非 x-vector)则 generate 会卡死/极慢
        → 统一 x-vector 模式 (仅说话人嵌入), 稳定且快; ref_text 参数保留但暂不使用"""
        self._ensure_model()
        ref = self._clip_ref_audio(ref_audio, ref_offset)
        self._validate_ref_audio(ref)
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref, ref_text="", x_vector_only_mode=True)
        return self._prompt

    def clone_synthesize(self, text, ref_audio: str,
                         ref_text: str | None = None,
                         language: str = "Chinese",
                         out_path: str | None = None):
        """声音克隆合成. 支持 str 或 list[str].
        ⚠️ 踩坑: 连续多次独立 generate 会卡死 → 必须批量; 但单批过大(如 67 句)解码序列过长
        也会极慢/卡死 → 内部按 BATCH_SIZE 分批, 每批一次 generate.
        返回: str -> (wav, sr); list -> (wavs_list, sr)"""
        import soundfile as sf
        self._ensure_model()
        if self._prompt is None:
            self.build_prompt(ref_audio, ref_text)
        gen_kwargs = self._gen_kwargs()
        if isinstance(text, str):
            wavs, sr = self.model.generate_voice_clone(
                text=text, language=language, voice_clone_prompt=self._prompt, **gen_kwargs)
            wav = wavs[0] if isinstance(wavs, list) else wavs
            if out_path:
                sf.write(out_path, wav, sr)
            return wav, sr
        # list: 分批合成 (每批一次 generate), 合并结果
        all_wavs: list = []
        sr = None
        total = len(text)
        for i in range(0, total, self.batch_size):
            batch = text[i:i + self.batch_size]
            wavs, sr = self.model.generate_voice_clone(
                text=batch, language=language, voice_clone_prompt=self._prompt, **gen_kwargs)
            all_wavs.extend(wavs if isinstance(wavs, list) else [wavs])
            print(f"[TTS] 批量 {i + 1}-{min(i + len(batch), total)}/{total} 完成", flush=True)
        return all_wavs, sr

    def _gen_kwargs(self) -> dict:
        """组装可选生成参数 (仅传已配置项, 其余用 qwen_tts 默认)"""
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs

    def release(self):
        """用完即释放显存"""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
