"""视频配音管道: 上传视频 → 音轨提取 → 人声分离 → ASR → 翻译中文 → 声音克隆 → 时间轴对齐 → 混流

Pipeline.run(video, ref_audio, ref_text) -> PipelineResult
模型串行"用完即释放"; 每步产物落盘到 workspace/job-*/ 支持断点续跑
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from app.audio import (extract_audio, match_loudness, mix_to_video,
                       probe_video_info, read_wav, time_stretch, trim_silence,
                       write_wav)
from app.engines.separator import get_separator
from app.engines.asr import ASREngine, Segment
from app.engines.translator import get_translator
from app.engines.tts import TTSEngine

WORKSPACE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")

# 时间轴对齐参数
STRETCH_MIN, STRETCH_MAX = 0.6, 1.8   # 变速范围 (中文通常比原文短, 需要拉伸)
OUT_SR = 44100


class RunLock:
    """全局任务互斥锁 (workspace/.run.lock): 防止多任务并发导致显存互踩/目录串扰

    锁文件含 pid+时间戳; 进程已死或超过 6 小时视为 stale 自动接管 (防崩溃残留卡死)"""

    LOCK_PATH = os.path.join(WORKSPACE_ROOT, ".run.lock")
    STALE_HOURS = 6

    def __init__(self):
        self._held = False

    def acquire(self):
        if os.path.exists(self.LOCK_PATH):
            if not self._is_stale():
                raise RuntimeError(
                    "检测到另一个任务正在运行 (并发互斥). 请等待其完成; "
                    "若确认没有任务在跑, 可手动删除 workspace/.run.lock")
            try:
                os.remove(self.LOCK_PATH)   # stale 锁: 删除后接管
            except OSError:
                pass
        try:
            fd = os.open(self.LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()} {time.time()}")
            self._held = True
        except FileExistsError:
            raise RuntimeError(
                "检测到另一个任务正在运行 (并发互斥). 请等待其完成; "
                "若确认没有任务在跑, 可手动删除 workspace/.run.lock")

    def _is_stale(self) -> bool:
        try:
            with open(self.LOCK_PATH, encoding="utf-8") as f:
                pid_s, ts_s = f.read().split()
            if time.time() - float(ts_s) > self.STALE_HOURS * 3600:
                return True
            try:
                os.kill(int(pid_s), 0)      # 进程存活?
                return False
            except OSError:
                return True                 # 进程已死 = 崩溃残留
        except Exception:
            return True                     # 读不了就当 stale

    def release(self):
        if self._held:
            try:
                os.remove(self.LOCK_PATH)
            except OSError:
                pass
            self._held = False


def _save_state(res: "PipelineResult", params: dict):
    """持久化断点状态: res 可序列化字段 + 运行参数 → workspace/state.json"""
    data = {
        "res": {
            "workspace": res.workspace, "video": res.video,
            "source_audio": res.source_audio, "vocals": res.vocals,
            "instrumental": res.instrumental,
            "asr_segments": [list(t) for t in res.asr_segments],
            "zh_lines": res.zh_lines,
            "dubbed_audio": res.dubbed_audio, "output_video": res.output_video,
            "timings": res.timings, "drift_seconds": res.drift_seconds,
            "volume_gain": res.volume_gain, "quality": res.quality,
            "stretch": list(res.stretch),
            "align_mode": res.align_mode,
            "mix_background": res.mix_background,
        },
        "params": params,
    }
    with open(os.path.join(res.workspace, "state.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_state(workspace: str, state_path: str) -> tuple["PipelineResult", dict]:
    """读取断点状态, 重建 PipelineResult + 上次运行参数"""
    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)
    r = data["res"]
    res = PipelineResult(
        workspace=r["workspace"], video=r["video"],
        source_audio=r["source_audio"], vocals=r["vocals"],
        instrumental=r["instrumental"],
        asr_segments=[tuple(t) for t in r["asr_segments"]],
        zh_lines=r["zh_lines"], dubbed_audio=r["dubbed_audio"],
        output_video=r["output_video"], timings=r["timings"],
        drift_seconds=r["drift_seconds"], volume_gain=r["volume_gain"],
        quality=r["quality"], stretch=tuple(r["stretch"]),
        align_mode=r.get("align_mode", "stretch"),
        mix_background=r.get("mix_background", False))
    return res, data["params"]


@dataclass
class PipelineResult:
    workspace: str
    video: str                  # 输入视频
    source_audio: str = ""      # 提取的音轨
    vocals: str = ""            # 分离人声
    instrumental: str = ""      # 分离伴奏
    asr_segments: list = field(default_factory=list)   # [(start, end, orig)]
    zh_lines: list = field(default_factory=list)       # 逐句中文
    dubbed_audio: str = ""      # 对齐后的配音音频
    output_video: str = ""      # 最终成品
    timings: dict = field(default_factory=dict)        # 每步耗时
    drift_seconds: float = 0.0  # 时间轴漂移量(兜底)
    volume_gain: float = 1.0    # 用户额外增益 (rerun_segment 时保持与主流程一致)
    quality: str = "copy"       # 混流画质档位 (rerun_segment 时保持)
    stretch: tuple = (STRETCH_MIN, STRETCH_MAX)  # 变速范围 (rerun_segment 时保持)
    align_mode: str = "stretch"  # 对齐模式: 'stretch' 拉伸填充 | 'natural' 自然语速居中
    mix_background: bool = False  # 混流时是否混合背景音(原伴奏)

    def summary(self) -> str:
        s = [f"视频: {self.video}", f"ASR 分段: {len(self.asr_segments)}", "---- 原文 / 译文 ----"]
        for (st, en, orig), zh in zip(self.asr_segments, self.zh_lines):
            s.append(f"  [{st:6.1f}-{en:6.1f}] {orig}")
            s.append(f"             {zh}")
        s.append(f"配音音频: {self.dubbed_audio}")
        s.append(f"成品: {self.output_video}")
        s.append(f"耗时: {', '.join(f'{k}={v:.0f}s' for k, v in self.timings.items())}")
        if self.drift_seconds > 1.0:
            s.append(f"⚠️ 时间轴漂移 {self.drift_seconds:.1f}s (部分句超出变速范围)")
        return "\n".join(s)


class Pipeline:
    # 阶段预估用时系数 (每视频秒, 固定, 每句): 粗糙线性模型, 供日志显示"预估用时"
    _EST_COEF = {
        "extract":   (0.02, 0.5, 0.0),   # ffmpeg 启动 + 轻量重编码
        "separate":  (0.25, 1.0, 0.0),   # MDX 约 4x 实时 (3060)
        "asr":       (0.28, 1.0, 0.0),   # faster-whisper turbo 约 3.5x 实时
        "translate": (0.0, 2.0, 1.5),    # API 每批 ~1.5s/句 (本地兜底更慢)
        "tts":       (0.0, 3.0, 2.5),    # 克隆合成 ~2.5s/句
        "align":     (0.0, 0.3, 0.1),    # 变速/拼接 ~0.1s/句
        "mix":       (0.02, 0.5, 0.0),   # 直通快, 压缩慢
    }

    def _est(self, key: str, video_dur: float, n_lines: int) -> float:
        """按规模估算阶段用时(秒); 下限 1s, 避免日志显示"预计 0s\""""
        a, b, c = self._EST_COEF.get(key, (0.0, 1.0, 0.0))
        return max(1.0, a * video_dur + b + c * n_lines)

    def __init__(self, workspace: str | None = None):
        self.workspace = workspace or os.path.join(
            WORKSPACE_ROOT, f"job-{time.strftime('%m%d-%H%M%S')}-{uuid4().hex[:6]}")
        os.makedirs(self.workspace, exist_ok=True)

    # ---------------- 主流程 (生成器: 每阶段 yield 中间状态, 供 UI 流式更新) ----------------
    def run_iter(self, video_path: str, ref_audio: str | None = None,
                 ref_text: str | None = None, target_lang: str = "Chinese",
                 source_lang: str | None = None,
                 stretch: tuple[float, float] = (STRETCH_MIN, STRETCH_MAX),
                 separator: str = "mdx23c", volume_gain: float = 1.0,
                 quality: str = "copy", audio_stream: int = 0,
                 batch_size: int | None = None,
                 align_mode: str = "stretch",
                 mix_background: bool = False,
                 num_overlap: int | None = None,
                 translate_config: dict | None = None,
                 asr_config: dict | None = None,
                 stop_after: int | None = None):
        """yield (stage, res, msg): stage 1-7, res 为逐步填充的中间结果, msg 为进度消息
        ref_audio=None 时抛错 (必须显式指定克隆音色; 自动用原视频人声模式已移除)
        volume_gain: 响度匹配后的额外增益 (1.0=仅自动匹配); quality: 'copy'|'balanced'|'small'
        audio_stream: 提取音轨时选择的音频流索引 (0 起, 多音轨视频用)
        batch_size: TTS 每批句数 (None → VP_TTS_BATCH_SIZE → 默认 12)
        align_mode: 对齐模式 'stretch'(拉伸填充) | 'natural'(自然语速居中)
        mix_background: 混流时是否混合背景音(原伴奏)
        num_overlap: 分离重叠数 (None → 配置/默认)
        translate_config: 翻译 API 配置 dict (api_key/base_url/model, 传给 get_translator)
        asr_config: ASR 配置 dict (initial_prompt/hotwords, 提升专有名词识别)
        stop_after: 运行到指定阶段完成后提前停止 (当前仅支持 4=完成翻译;
        停止后 state.json 保留译文, 可在 UI 改译文后点「续跑上次」从阶段5继续)
        断点续跑: 同一 workspace 已存在 state.json 时自动重建状态并跳过已完成阶段;
        并发互斥: 全程持有全局锁, 另一任务运行中会抛错"""
        lock = RunLock()
        lock.acquire()
        try:
            yield from self._run_iter_locked(
                video_path, ref_audio, ref_text, target_lang, source_lang,
                stretch, separator, volume_gain, quality, audio_stream, batch_size,
                align_mode, mix_background, num_overlap, translate_config, asr_config,
                stop_after)
        finally:
            lock.release()      # 正常/异常/取消(GeneratorExit)均释放

    def _run_iter_locked(self, video_path: str, ref_audio: str | None = None,
                         ref_text: str | None = None, target_lang: str = "Chinese",
                         source_lang: str | None = None,
                         stretch: tuple[float, float] = (STRETCH_MIN, STRETCH_MAX),
                         separator: str = "mdx23c", volume_gain: float = 1.0,
                         quality: str = "copy", audio_stream: int = 0,
                         batch_size: int | None = None,
                         align_mode: str = "stretch",
                         mix_background: bool = False,
                         num_overlap: int | None = None,
                         translate_config: dict | None = None,
                         asr_config: dict | None = None,
                         stop_after: int | None = None):
        params = dict(video_path=video_path, ref_audio=ref_audio, ref_text=ref_text,
                      target_lang=target_lang, source_lang=source_lang,
                      separator=separator, volume_gain=volume_gain, quality=quality,
                      stretch=list(stretch), audio_stream=audio_stream,
                      batch_size=batch_size, align_mode=align_mode,
                      mix_background=mix_background, num_overlap=num_overlap,
                      translate_config=translate_config, asr_config=asr_config)
        state_path = os.path.join(self.workspace, "state.json")
        if os.path.isfile(state_path):
            # 断点续跑: 重建 res 并沿用上次参数 (产物与参数一致, 忽略本次传入)
            res, saved = _load_state(self.workspace, state_path)
            params = saved
            video_path, ref_audio, ref_text = (params["video_path"], params["ref_audio"],
                                               params["ref_text"])
            target_lang = params["target_lang"]
            source_lang = params["source_lang"]
            separator, volume_gain, quality = (params["separator"], params["volume_gain"],
                                               params["quality"])
            stretch = tuple(params["stretch"])
            audio_stream, batch_size = params["audio_stream"], params["batch_size"]
            align_mode = params.get("align_mode", "stretch")   # 兼容旧 state.json
            mix_background = params.get("mix_background", False)
            num_overlap = params.get("num_overlap")
            translate_config = params.get("translate_config")
            asr_config = params.get("asr_config")
            smin, smax = stretch
            yield 1, res, "检测到断点状态, 从上次中断处续跑"
        else:
            res = PipelineResult(workspace=self.workspace, video=video_path)
            res.volume_gain = volume_gain      # 记录参数供 rerun_segment 复用
            res.quality = quality
            res.stretch = stretch
            res.align_mode = align_mode
            res.mix_background = mix_background
            smin, smax = stretch
            if source_lang is None:
                source_lang = os.environ.get("VP_ASR_LANG")   # 新闻场景可配 en, 免自动检测
        t0 = time.time()
        info = probe_video_info(video_path)
        if not info:
            raise RuntimeError(f"无法解析视频: {video_path}")
        duration = float(info.get("duration") or 0)

        # 1. 音轨提取 (断点续跑: 产物已存在则跳过)
        if res.source_audio and os.path.isfile(res.source_audio):
            yield 1, res, "检测到 01_source.wav, 跳过音轨提取 (断点续跑)"
        else:
            t = time.time()
            yield 1, res, f"正在提取音轨（预计 {self._est('extract', duration, 0):.0f} 秒）..."
            res.source_audio = os.path.join(self.workspace, "01_source.wav")
            extract_audio(video_path, res.source_audio, audio_stream=audio_stream)
            res.timings["extract"] = time.time() - t
            _save_state(res, params)
            yield 1, res, (f"音轨提取 → 01_source.wav ({duration:.0f}s 视频)"
                           f" ｜实际 {res.timings['extract']:.0f} 秒")

        # 2. 人声分离 (MDX23C / UVR-MDX onnx / Kim_Vocal_2 可选; 断点续跑: 产物已存在则跳过)
        if res.vocals and os.path.isfile(res.vocals):
            yield 2, res, "检测到 02_vocals.wav, 跳过人声分离 (断点续跑)"
        else:
            t = time.time()
            yield 2, res, f"正在运行人声分离（预计 {self._est('separate', duration, 0):.0f} 秒）..."
            wave, sr = read_wav(res.source_audio)
            sep = get_separator(separator)
            if num_overlap:
                sep.cfg["num_overlap"] = num_overlap   # UI 参数覆盖 (4/8/16)
            try:
                vocals, inst = sep.separate(wave, sr)
            finally:
                # 异常/取消时也保证释放
                del sep
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            res.vocals = os.path.join(self.workspace, "02_vocals.wav")
            res.instrumental = os.path.join(self.workspace, "02_inst.wav")
            write_wav(vocals, 44100, res.vocals)
            write_wav(inst, 44100, res.instrumental)
            res.timings["separate"] = time.time() - t
            _save_state(res, params)
            yield 2, res, (f"人声分离完成 ({separator}) → 02_vocals.wav / 02_inst.wav"
                           f" ｜实际 {res.timings['separate']:.0f} 秒")

        # 参考音色必须由用户指定 (名人库或上传音频); 自动用原视频人声模式已移除
        if not ref_audio:
            raise ValueError("请选择配音音色 (上传音频或名人音色库)")

        # 3. ASR (断点续跑: 已有识别结果则跳过)
        if res.asr_segments:
            segments = [Segment(s, e, t) for s, e, t in res.asr_segments]
            yield 3, res, f"检测到 ASR 结果 {len(segments)} 段, 跳过识别 (断点续跑)"
        else:
            t = time.time()
            yield 3, res, f"正在语音识别（预计 {self._est('asr', duration, 0):.0f} 秒）..."
            asr = ASREngine(**(asr_config or {}))
            try:
                segments = asr.transcribe(res.vocals, language=source_lang)
            finally:
                asr.release()   # 异常/取消时也保证释放
            res.timings["asr"] = time.time() - t
            res.asr_segments = [(s.start, s.end, s.text) for s in segments]
            _save_state(res, params)
            yield 3, res, f"ASR 完成: {len(segments)} 段 ｜实际 {res.timings['asr']:.0f} 秒"

        # 4. 翻译 (DeepSeek 分批+行数校验, 保证逐句对齐; 断点续跑: 已有译文则跳过)
        if res.zh_lines:
            yield 4, res, f"检测到译文 {len(res.zh_lines)} 行, 跳过翻译 (断点续跑)"
        else:
            t = time.time()
            yield 4, res, f"正在翻译（预计 {self._est('translate', duration, len(segments)):.0f} 秒）..."
            translator = get_translator(**(translate_config or {}))
            orig_lines = [seg.text for seg in segments]
            res.zh_lines = translator.translate_lines(orig_lines, target_lang)
            res.timings["translate"] = time.time() - t
            _save_state(res, params)
            yield 4, res, f"翻译完成: {len(res.zh_lines)} 行 ｜实际 {res.timings['translate']:.0f} 秒"

        # 4.5 提前停止: 仅执行到翻译完成 (stop_after=4, 不配音/不对齐/不混流)
        if stop_after == 4:
            res.timings["total"] = time.time() - t0
            _save_state(res, params)
            yield 4, res, ("✅ 已停止于「翻译完成」阶段：未执行克隆配音/对齐/混流。"
                           "可在 ②字幕编辑 修改译文后，点「↻ 续跑上次」继续配音。")
            return

        # 5. 声音克隆合成 (分批: qwen_tts 连续独立调用会卡死, 单批过大也极慢; 每批 yield 进度)
        # 断点续跑: 03_zh/ 已有的句跳过, 只补缺合成 (TTS 中途失败续跑不重算已完成句)
        t = time.time()
        seg_dir = os.path.join(self.workspace, "03_zh")     # 每句配音落盘, 支持逐句重跑
        os.makedirs(seg_dir, exist_ok=True)
        active = [(i, zh) for i, zh in enumerate(res.zh_lines) if zh.strip()]
        zh_wavs: list = [None] * len(res.zh_lines)
        total = len(active)
        missing: list[tuple[int, str]] = []
        for i, zh in active:
            p = os.path.join(seg_dir, f"{i}.wav")
            if os.path.isfile(p):
                w, sr = read_wav(p)          # 读回 (2,L) → 取单声道, 与 TTS 输出一致
                zh_wavs[i] = (w[0] if w.ndim == 2 else w, sr)
            else:
                missing.append((i, zh))
        tts = TTSEngine(batch_size=batch_size)
        try:
            if missing:
                yield 5, res, f"正在克隆配音（预计 {self._est('tts', duration, total):.0f} 秒）..."
                yield 5, res, "正在加载 TTS 模型 (1.7B, 首次约 1-2 分钟, 请稍候)..."
                tts.build_prompt(ref_audio, ref_text)
                for start in range(0, len(missing), tts.batch_size):
                    batch = missing[start:start + tts.batch_size]
                    wavs, wav_sr = tts.clone_synthesize(
                        [zh for _, zh in batch], ref_audio, ref_text, language=target_lang)
                    for (i, _), w in zip(batch, wavs):
                        zh_wavs[i] = (w, wav_sr)
                        write_wav(np.asarray(w, dtype=np.float32), wav_sr,
                                  os.path.join(seg_dir, f"{i}.wav"))
                    done = min(start + len(batch), total)
                    yield 5, res, f"TTS 批量 {done}/{total} 完成"
        finally:
            tts.release()   # 异常/取消时也保证释放
        res.timings["tts"] = time.time() - t
        if missing:
            _save_state(res, params)
            yield 5, res, (f"克隆配音完成: {total} 句 (已落盘 03_zh/)"
                           f" ｜实际 {res.timings['tts']:.0f} 秒")
        else:
            yield 5, res, f"检测到 03_zh/ 全部 {total} 句, 跳过克隆配音 (断点续跑)"

        # 6. 时间轴对齐拼接 (断点续跑: 已对齐则跳过 6 + 6.3)
        if res.dubbed_audio and os.path.isfile(res.dubbed_audio):
            yield 6, res, "检测到 06_dubbed.wav, 跳过对齐+响度 (断点续跑)"
        else:
            t = time.time()
            yield 6, res, f"正在时间轴对齐（预计 {self._est('align', duration, len(segments)):.0f} 秒）..."
            dubbed, drift = self._align(segments, zh_wavs, duration, smin, smax,
                                        mode=align_mode)
            res.drift_seconds = drift
            res.dubbed_audio = os.path.join(self.workspace, "06_dubbed.wav")
            write_wav(dubbed, OUT_SR, res.dubbed_audio)
            res.timings["align"] = time.time() - t
            yield 6, res, (f"时间轴对齐完成: {len(dubbed[0])/OUT_SR:.1f}s 配音 (漂移 {drift:.1f}s)"
                           f" ｜实际 {res.timings['align']:.0f} 秒")

            # 6.3 响度匹配: 配音增益到与原音轨相近, 再乘用户额外增益
            try:
                src_w, src_sr = read_wav(res.source_audio)
                dub_w, dub_sr = read_wav(res.dubbed_audio)
                dub_w = match_loudness(src_w, dub_w) * float(volume_gain)
                write_wav(dub_w, OUT_SR, res.dubbed_audio)
                yield 6, res, "响度匹配完成 (自动对齐原片音量)"
            except Exception:
                pass

        # 6.5 导出 SRT (原文/译文; 产物缺失时补写)
        srt_ok = (os.path.isfile(os.path.join(self.workspace, "03_orig.srt"))
                  and os.path.isfile(os.path.join(self.workspace, "03_zh.srt")))
        if not srt_ok:
            self._write_srt(res)
        _save_state(res, params)

        # 7. 混流回视频 (默认视频流直通, 可选压缩档位; 断点续跑: 成品已存在则跳过)
        if res.output_video and os.path.isfile(res.output_video):
            yield 7, res, "成品已存在, 跳过混流 (断点续跑)"
        else:
            t = time.time()
            yield 7, res, f"正在混流出片（预计 {self._est('mix', duration, 0):.0f} 秒）..."
            res.output_video = os.path.join(self.workspace, "07_output.mp4")
            mix_to_video(video_path, res.dubbed_audio, res.output_video,
                         quality=quality,
                         background_wav=(res.instrumental if mix_background else None))
            res.timings["mix"] = time.time() - t
            _save_state(res, params)
            yield 7, res, (f"混流出片 → 07_output.mp4"
                           f" ｜实际 {res.timings['mix']:.0f} 秒")

        res.timings["total"] = time.time() - t0
        _save_state(res, params)
        yield 7, res, f"✅ 全部完成 · 耗时 {res.timings['total']:.0f} 秒"

    # ---------------- 主流程 (同步包装, 兼容命令行/旧调用) ----------------
    def run(self, video_path: str, ref_audio: str | None = None, ref_text: str | None = None,
            target_lang: str = "Chinese", source_lang: str | None = None,
            stretch: tuple[float, float] = (STRETCH_MIN, STRETCH_MAX),
            separator: str = "mdx23c", volume_gain: float = 1.0,
            quality: str = "copy", audio_stream: int = 0,
            batch_size: int | None = None,
            align_mode: str = "stretch", mix_background: bool = False,
            num_overlap: int | None = None, translate_config: dict | None = None,
            asr_config: dict | None = None, stop_after: int | None = None,
            on_step=None) -> PipelineResult:
        """同步执行管道 (stop_after=4 时只到翻译完成), 返回最终 PipelineResult.
        on_step(stage, msg): 每阶段进度回调 (兼容旧接口)"""
        res: PipelineResult | None = None
        for stage, r, msg in self.run_iter(video_path, ref_audio, ref_text,
                                           target_lang, source_lang, stretch,
                                           separator, volume_gain, quality,
                                           audio_stream, batch_size, align_mode,
                                           mix_background, num_overlap, translate_config,
                                           asr_config, stop_after):
            res = r
            if on_step:
                on_step(stage, msg)
            elif msg:
                print(f"[{stage}/7] {msg}")
        return res

    # ---------------- 增量重跑 (UI 逐句编辑用) ----------------
    def rerun_segment(self, res: PipelineResult, idx: int, new_zh: str,
                      ref_audio: str | None = None, ref_text: str | None = None,
                      target_lang: str = "Chinese",
                      stretch: tuple[float, float] | None = None) -> tuple[PipelineResult, str]:
        """只重合成第 idx 句译文并重新对齐+混流 (不重跑分离/ASR/翻译). 返回 (res, 状态消息)
        stretch=None 时沿用主流程 res.stretch; 响度/画质沿用 res.volume_gain / res.quality"""
        if not res or idx < 0 or idx >= len(res.asr_segments):
            return res, "无有效任务或索引越界"
        new_zh = new_zh.strip()
        if not new_zh:
            return res, "译文为空, 已跳过"
        res.zh_lines[idx] = new_zh
        segments = [Segment(start=s[0], end=s[1], text=s[2]) for s in res.asr_segments]
        # 参考音色必须由用户指定; 自动用原视频人声模式已移除
        if not ref_audio:
            raise ValueError("请选择配音音色 (上传音频或名人音色库)")
        seg_dir = os.path.join(self.workspace, "03_zh")

        # 仅合成该句 (批量列表包装, 避免连续 generate 卡死)
        tts = TTSEngine()
        try:
            tts.build_prompt(ref_audio, ref_text)
            wavs, wav_sr = tts.clone_synthesize([new_zh], ref_audio, ref_text, language=target_lang)
        finally:
            tts.release()   # 异常时也保证释放
        os.makedirs(seg_dir, exist_ok=True)
        write_wav(np.asarray(wavs[0], dtype=np.float32), wav_sr,
                  os.path.join(seg_dir, f"{idx}.wav"))

        # 重读全部句 wav, 全量重新对齐 (CPU 快, 无需 GPU)
        zh_wavs = []
        for i in range(len(segments)):
            p = os.path.join(seg_dir, f"{i}.wav")
            if os.path.exists(p):
                w, sr = read_wav(p)
                zh_wavs.append((w, sr))
            else:
                zh_wavs.append(None)
        duration = float(probe_video_info(res.video).get("duration") or 0)
        smin, smax = stretch if stretch is not None else res.stretch
        dubbed, drift = self._align(segments, zh_wavs, duration, smin, smax,
                                    mode=getattr(res, "align_mode", "stretch"))
        res.drift_seconds = drift
        write_wav(dubbed, OUT_SR, res.dubbed_audio)
        # 响度匹配保持一致 (沿用主流程的用户额外增益)
        try:
            src_w, src_sr = read_wav(res.source_audio)
            dub_w, _ = read_wav(res.dubbed_audio)
            write_wav(match_loudness(src_w, dub_w) * res.volume_gain, OUT_SR, res.dubbed_audio)
        except Exception:
            pass
        mix_to_video(res.video, res.dubbed_audio, res.output_video, quality=res.quality,
                     background_wav=(res.instrumental if getattr(res, "mix_background", False)
                                     else None))
        self._write_srt(res)
        return res, f"第 {idx + 1} 句已重新合成并混流 (漂移 {drift:.1f}s)"

    def _write_srt(self, res: PipelineResult):
        """导出 03_orig.srt / 03_zh.srt (原文/译文)"""
        import pysubs2
        subs_orig, subs_zh = pysubs2.SSAFile(), pysubs2.SSAFile()
        for i, (st, en, orig) in enumerate(res.asr_segments):
            subs_orig.append(pysubs2.SSAEvent(start=int(st * 1000), end=int(en * 1000), text=orig))
            zh = res.zh_lines[i] if i < len(res.zh_lines) else ""
            subs_zh.append(pysubs2.SSAEvent(start=int(st * 1000), end=int(en * 1000), text=zh))
        subs_orig.save(os.path.join(self.workspace, "03_orig.srt"))
        subs_zh.save(os.path.join(self.workspace, "03_zh.srt"))

    # ---------------- 时间轴对齐 ----------------
    def _align(self, segments: list[Segment], zh_wavs: list,
               video_duration: float, smin: float = STRETCH_MIN,
               smax: float = STRETCH_MAX,
               mode: str = "stretch") -> tuple[np.ndarray, float]:
        """逐句对齐原字幕窗并拼接. 返回 (stereo_wave, 累计漂移)

        mode='stretch': 变速到窗长 (拉伸/压缩填充时间轴).
        mode='natural': 短句不变速, 窗内中点对齐(前后对称空隙); 长句仍压缩填充.
        方案A (窗对齐兜底, 非 natural 短句): 变速后仍短于窗 → 尾部补静音(上限1.5s);
                           仍超窗 ≤0.3s → 裁剪尾部到窗长. 消除空隙与轻微溢出.
        方案B (间隙吸收回退): 空隙(carry)被后续被推移的句子借用提前,
                           末句回到原时间轴, 长视频尾部不再累积滞后."""
        total_samples = int((video_duration + 1.0) * OUT_SR)
        out = np.zeros((2, total_samples), dtype=np.float32)
        pos = 0.0                                   # 当前位置(秒)
        drift = 0.0
        carry = 0.0                                 # 方案B: 前面空隙累计的可回退量(秒)
        for seg, zh in zip(segments, zh_wavs):
            if zh is None:
                continue
            start, end = seg.start, seg.end
            wav, wav_sr = zh
            wav = trim_silence(wav, wav_sr)     # 修剪前后静音, 贴合更准
            target_len = end - start
            actual = len(wav) / wav_sr
            is_short = mode == "natural" and target_len > 0.05 and actual < target_len
            if not is_short:
                # librosa time_stretch: rate>1 加速(输出变短). ratio = actual/target
                ratio = actual / target_len if target_len > 0.05 else 1.0
                ratio = float(np.clip(ratio, 1.0 / smax, 1.0 / smin))
                if ratio < 0.99 or ratio > 1.01:
                    wav = time_stretch(wav, ratio, wav_sr)
                    actual = len(wav) / wav_sr
            # ---- 方案A: 窗对齐兜底 (natural 短句跳过, 保持自然语速) ----
            if target_len > 0.05 and not is_short:
                if actual < target_len:
                    pad = min(target_len - actual, 1.5)   # 最多补 1.5s 静音, 超出留空隙
                    if pad > 0.02:
                        wav = np.pad(wav, (0, int(round(pad * wav_sr))))
                        actual += pad
                elif actual - target_len <= 0.3:
                    wav = wav[:int(round(target_len * wav_sr))]   # 微超: 裁剪尾部
                    actual = target_len
            # ---- natural 短句: 窗内中点对齐 (anchor 为目标放置点) ----
            anchor = start
            if is_short:
                anchor = start + (target_len - actual) / 2.0
            # 重采样到输出采样率
            if wav_sr != OUT_SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=OUT_SR)
            if wav.ndim == 1:
                wav = np.stack([wav, wav])
            # ---- 方案B: 间隙吸收回退 (以 anchor 为目标位置) ----
            if pos < anchor:
                carry += anchor - pos             # 本句前有空隙: 累计为可回退量
                start_pos = anchor
            else:
                start_pos = max(anchor, pos - carry)   # 被推移时借空隙提前
                carry = max(0.0, carry - (pos - start_pos))
            end_pos = start_pos + len(wav[0]) / OUT_SR
            if end_pos > video_duration + 1.0:
                break
            si = int(start_pos * OUT_SR)
            out[:, si:si + wav.shape[1]] = wav[:, :out.shape[1] - si]
            if start_pos > anchor + 0.05:
                drift += start_pos - anchor
            pos = end_pos
        return out, drift
