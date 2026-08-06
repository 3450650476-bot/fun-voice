"""ffmpeg 音频/视频工具 (使用备份的 models\ffmpeg\bin)

功能: 音轨提取 / 变速不变调 / 静音修剪 / 混流回视频
"""
from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()   # 保证 .env 中 FFMPEG_PATH / VP_MODELS 生效 (与 translator.py 同款)

# 默认路径相对项目根定位 (K:\视频翻译与配音\models), 可被 VP_MODELS / FFMPEG_PATH 覆盖
_MODELS_ROOT = os.environ.get("VP_MODELS", str(Path(__file__).resolve().parents[2] / "models"))
FFMPEG_PATH = os.environ.get(
    "FFMPEG_PATH",
    str(Path(_MODELS_ROOT) / "ffmpeg" / "bin" / "ffmpeg.exe"),
)
FFPROBE_PATH = os.path.join(os.path.dirname(FFMPEG_PATH), "ffprobe.exe")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """运行 ffmpeg 命令, 失败抛错"""
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(cmd[:8])}...\n{proc.stderr[-800:]}")
    return proc


def probe_audio_streams(path: str) -> int:
    """返回音频流数量. 文件无法解析时抛 RuntimeError"""
    proc = subprocess.run(
        [FFPROBE_PATH, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    if proc.returncode != 0:
        raise RuntimeError(f"无法解析视频 {os.path.basename(path)}: {proc.stderr.strip()[-300:]}")
    return len([ln for ln in proc.stdout.splitlines() if ln.strip()])


def extract_audio(video_path: str, out_wav: str, sr: int = 44100,
                  audio_stream: int = 0) -> str:
    """提取视频音轨为 wav (pcm_s16le, 双声道, 目标采样率)

    audio_stream: 选择第 N 条音频流 (0 起), 多音轨视频可指定
    无音轨 / 流索引越界 / 提取产物为空时报错 (中文友好提示, 供 UI 直接展示)"""
    n_streams = probe_audio_streams(video_path)
    if n_streams == 0:
        raise ValueError("视频没有音轨, 无法提取 (该视频可能只有画面)")
    if audio_stream >= n_streams:
        raise ValueError(f"视频只有 {n_streams} 条音轨, 无法选择第 {audio_stream + 1} 条")
    _run([FFMPEG_PATH, "-y", "-i", video_path, "-vn",
          "-map", f"0:a:{audio_stream}",
          "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "2", out_wav])
    if not os.path.isfile(out_wav) or os.path.getsize(out_wav) <= 44:
        raise ValueError(f"音轨提取失败: 产物 {out_wav} 为空 (输出异常)")
    return out_wav


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 wav -> (wave (C,L) float32, sr)"""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        data = np.stack([data, data])
    else:
        data = data.T
    return data, sr


def write_wav(wave: np.ndarray, sr: int, out_path: str):
    sf.write(out_path, wave.T, sr)


def time_stretch(wave: np.ndarray, ratio: float, sr: int) -> np.ndarray:
    """变速不变调: ratio>1 加速, ratio<1 减速 (librosa 相位声码器)"""
    import librosa
    if abs(ratio - 1.0) < 0.02:
        return wave
    # librosa 沿时间轴拉伸: time_stretch(wave, rate) rate>1 加速
    stretched = librosa.effects.time_stretch(wave, rate=ratio)
    return stretched


def trim_silence(wave: np.ndarray, sr: int, top_db: float = 40) -> np.ndarray:
    """修剪首尾静音"""
    import librosa
    idx = librosa.effects.trim(wave[0] if wave.ndim == 2 else wave,
                               top_db=top_db, frame_length=1024, hop_length=256)
    if wave.ndim == 2:
        return wave[:, idx[1][0]:idx[1][1]]
    return wave[idx[1][0]:idx[1][1]]


@lru_cache(maxsize=64)
def probe_duration(path: str) -> float:
    """获取媒体时长(秒). 同一路径结果缓存 (上传后文件不变, 缓存安全)"""
    proc = subprocess.run(
        [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise RuntimeError(f"无法探测时长: {path}")


def match_loudness(ref_wave: np.ndarray, dub_wave: np.ndarray,
                   max_gain: float = 4.0) -> np.ndarray:
    """响度匹配: 把配音增益到与原音轨相近的 RMS 响度 (线性增益, 不改动态)

    max_gain: 最大增益倍数, 防止静音参考导致爆音
    """
    def _rms(w):
        return float(np.sqrt(np.mean(np.asarray(w, dtype=np.float64) ** 2)))
    r_ref = _rms(ref_wave)
    r_dub = _rms(dub_wave)
    if r_ref < 1e-6 or r_dub < 1e-6:
        return dub_wave
    gain = min(r_ref / r_dub, max_gain)
    return np.asarray(dub_wave, dtype=np.float32) * float(gain)


@lru_cache(maxsize=64)
def probe_video_info(path: str) -> dict:
    """获取视频基本信息: 时长/分辨率/编码/体积. 同一路径结果缓存"""
    info: dict = {}
    try:
        proc = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height:format=duration,size",
             "-of", "json", path],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        import json
        data = json.loads(proc.stdout or "{}")
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                info["video_codec"] = s.get("codec_name")
                info["width"], info["height"] = s.get("width"), s.get("height")
            elif s.get("codec_type") == "audio":
                info.setdefault("audio_codec", s.get("codec_name"))
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0) or 0)
        info["size_mb"] = float(fmt.get("size", 0) or 0) / 1048576
    except Exception:
        pass
    return info


def mix_to_video(video_path: str, audio_wav: str, out_path: str,
                 quality: str = "copy",
                 background_wav: str | None = None) -> str:
    """将新音轨混流回视频.

    quality: 'copy' 视频流无损直通(画质最佳, 体积=源视频, 默认)
             'balanced' h264 crf 23 (体积约减 30-50%)
             'small'     h264 crf 28 (体积约减 50-70%)
    background_wav: 非空时把背景音(如原伴奏)以 0.35 音量混合进音轨 (amix, 以配音时长为准)
    输出以视频完整时长为准 (-t, 音频不足部分静音而非截断画面);
    +faststart 使 moov 前置, 网页播放器边下边播 (秒开).
    """
    # 视频时长 (lru_cache, 不额外起 ffprobe); 探测失败时退化为 -shortest
    dur = 0.0
    info = probe_video_info(video_path)
    if info:
        try:
            dur = float(info.get("duration") or 0)
        except (TypeError, ValueError):
            dur = 0.0
    tail = (["-t", f"{dur:.3f}"] if dur > 0 else ["-shortest"]) + \
           ["-movflags", "+faststart"]
    base = [FFMPEG_PATH, "-y", "-i", video_path, "-i", audio_wav]
    if background_wav:
        base += ["-i", background_wav]
    if background_wav:
        # 背景音压低到 0.35 后与配音混合 (amix normalize=0 保持配音响度; duration=first 以配音为准)
        afilter = ("[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];"
                   "[2:a]aformat=sample_rates=44100:channel_layouts=stereo,"
                   "volume=0.35[a2];"
                   "[a1][a2]amix=inputs=2:duration=first:normalize=0[aout]")
        map_audio = ["-filter_complex", afilter, "-map", "[aout]"]
    else:
        map_audio = ["-map", "1:a:0"]

    def run_with(vcodec: list):
        cmd = base + ["-map", "0:v:0"] + map_audio + vcodec + \
              ["-c:a", "aac", "-b:a", "192k", *tail, out_path]
        _run(cmd)

    if quality == "copy":
        # 直通失败(容器/编码不兼容)时自动回退 crf 23
        try:
            run_with(["-c:v", "copy"])
            return out_path
        except RuntimeError:
            print("[混流] 容器/编码不兼容, 视频流已回退重编码 (crf 23, 画质接近无损)",
                  flush=True)
    crf = "23" if quality in ("copy", "balanced") else "28"
    run_with(["-c:v", "libx264", "-crf", crf, "-preset", "medium"])
    return out_path
