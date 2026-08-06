"""断点续跑 + 并发互斥测试 (fake 引擎 + 真实 soundfile 落盘产物)

覆盖: RunLock 互斥/stale 接管 / 全量跑完后续跑全跳过 / TTS 中途失败续跑只补缺
"""
import os
import shutil
import sys
import time
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.pipeline as P
from app.engines.asr import Segment


def _fake_extract(video_path, out_wav, audio_stream=0):
    """模拟 ffmpeg 提取: 真实写一个 wav 产物"""
    sf.write(out_wav, np.zeros((2, 44100), np.float32).T, 44100)
    return out_wav


def _fake_mix(src, dub, out, **k):
    """模拟混流: 写占位产物"""
    with open(out, "wb") as f:
        f.write(b"x")
    return out


def _align_stub(self, segs, wavs, dur, smin, smax, mode="stretch"):
    return np.zeros((2, int(dur * 44100) + 44100), np.float32), 0.0


class FakeSep:
    def separate(self, wave, sr):
        return np.zeros((2, 44100), np.float32), np.zeros((2, 44100), np.float32)


class FakeASR:
    def transcribe(self, *a, **k):
        return [Segment(float(i), float(i) + 1.0, f"seg{i}") for i in range(30)]

    def release(self):
        pass


class FakeTrans:
    def translate_lines(self, lines, target_lang):
        return [f"译:{t}" for t in lines]


class FakeTTS:
    def __init__(self, **kw):
        self.batch_size = kw.get("batch_size") or 12

    def build_prompt(self, *a, **k):
        pass

    def clone_synthesize(self, texts, *a, **k):
        return [np.zeros(16000, np.float32) for _ in texts], 16000

    def release(self):
        pass


class BoomTTS(FakeTTS):
    """第 2 批开始抛异常, 模拟 TTS 中途失败"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def clone_synthesize(self, texts, *a, **k):
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("tts boom")
        return super().clone_synthesize(texts, *a, **k)


class TestRunLock(unittest.TestCase):
    def setUp(self):
        self.lock_path = P.RunLock.LOCK_PATH
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)

    def tearDown(self):
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)

    def test_second_acquire_raises(self):
        l = P.RunLock()
        l.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "互斥"):
                P.RunLock().acquire()
        finally:
            l.release()

    def test_release_allows_reacquire(self):
        l = P.RunLock()
        l.acquire()
        l.release()
        l2 = P.RunLock()
        l2.acquire()
        l2.release()

    def test_stale_lock_taken_over(self):
        with open(self.lock_path, "w", encoding="utf-8") as f:
            f.write("999999 0.0")          # 不存在的 pid + 旧时间戳
        l = P.RunLock()
        l.acquire()                        # stale 自动接管
        l.release()

    def test_live_lock_rejected(self):
        with open(self.lock_path, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {time.time()}")
        with self.assertRaisesRegex(RuntimeError, "互斥"):
            P.RunLock().acquire()


class PipelineResumeTest(unittest.TestCase):
    def setUp(self):
        self._patchers = [
            mock.patch.object(P, "extract_audio", _fake_extract),
            mock.patch.object(P, "probe_video_info", lambda p: {"duration": 10.0}),
            mock.patch.object(P, "get_separator", lambda *a, **k: FakeSep()),
            mock.patch.object(P, "ASREngine", FakeASR),
            mock.patch.object(P, "get_translator", lambda **k: FakeTrans()),
            mock.patch.object(P, "TTSEngine", FakeTTS),
            mock.patch.object(P, "match_loudness", lambda a, b: np.asarray(b, np.float32)),
            mock.patch.object(P, "mix_to_video", _fake_mix),
            mock.patch.object(P.Pipeline, "_align", _align_stub),
        ]
        for p in self._patchers:
            p.start()
        self.work = os.path.join("workspace", "_test_resume")
        shutil.rmtree(self.work, ignore_errors=True)
        if os.path.exists(P.RunLock.LOCK_PATH):
            os.remove(P.RunLock.LOCK_PATH)

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        shutil.rmtree(self.work, ignore_errors=True)
        if os.path.exists(P.RunLock.LOCK_PATH):
            os.remove(P.RunLock.LOCK_PATH)

    def test_full_run_then_resume_skips_everything(self):
        p1 = P.Pipeline(workspace=self.work)
        seq1 = [s for s, r, m in p1.run_iter("v.mp4", ref_audio="ref.wav",
                                             target_lang="Chinese")]
        self.assertEqual(seq1, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7])   # 30 句 → 3 批 TTS
        self.assertTrue(os.path.isfile(os.path.join(self.work, "state.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.work, "07_output.mp4")))

        # 续跑: 同一 workspace, 全部阶段跳过, 沿用保存的参数 (video/ref 传空占位)
        p2 = P.Pipeline(workspace=self.work)
        items2 = list(p2.run_iter("", "", target_lang="Chinese"))
        msgs2 = [m for s, r, m in items2]
        joined = "\n".join(msgs2)
        self.assertIn("检测到断点状态", joined)
        for kw in ("跳过音轨提取", "跳过人声分离", "跳过识别", "跳过翻译",
                   "跳过克隆配音", "跳过对齐", "跳过混流"):
            self.assertIn(kw, joined, kw)
        self.assertTrue(msgs2[-1].startswith("✅ 全部完成"))
        # 续跑沿用了原 video/ref (从 state.json), 产物完整
        res2 = items2[-1][1]
        self.assertEqual(res2.video, "v.mp4")
        self.assertEqual(res2.output_video, os.path.join(self.work, "07_output.mp4"))

    def test_tts_partial_failure_resume_only_missing(self):
        # 第一轮: TTS 第 2 批抛异常 → 前 12 句已落盘, 任务失败
        with mock.patch.object(P, "TTSEngine", BoomTTS):
            p1 = P.Pipeline(workspace=self.work)
            with self.assertRaisesRegex(RuntimeError, "tts boom"):
                list(p1.run_iter("v.mp4", ref_audio="ref.wav", target_lang="Chinese"))
        seg_dir = os.path.join(self.work, "03_zh")
        done_first = len([f for f in os.listdir(seg_dir) if f.endswith(".wav")])
        self.assertEqual(done_first, 12)          # 第 1 批 12 句落盘
        self.assertTrue(os.path.isfile(os.path.join(self.work, "state.json")))
        self.assertFalse(os.path.isfile(os.path.join(self.work, "06_dubbed.wav")))

        # 第二轮: 只补缺剩余 18 句, 不重算已完成的 12 句
        synth = {"n": 0}
        class TrackTTS(FakeTTS):
            def clone_synthesize(self, texts, *a, **k):
                synth["n"] += len(texts)
                return super().clone_synthesize(texts, *a, **k)
        with mock.patch.object(P, "TTSEngine", TrackTTS):
            p2 = P.Pipeline(workspace=self.work)
            msgs2 = [m for s, r, m in p2.run_iter("", "", target_lang="Chinese")]
        self.assertEqual(synth["n"], 18)          # 只补缺
        self.assertTrue(msgs2[-1].startswith("✅ 全部完成"))
        self.assertTrue(os.path.isfile(os.path.join(self.work, "06_dubbed.wav")))
        self.assertTrue(os.path.isfile(os.path.join(self.work, "07_output.mp4")))

    def test_concurrent_run_iter_rejected(self):
        p1 = P.Pipeline(workspace=self.work)
        it1 = p1.run_iter("v.mp4", ref_audio="ref.wav", target_lang="Chinese")
        next(it1)                                  # 首个 next 已持有全局锁
        p2 = P.Pipeline(workspace=self.work)
        with self.assertRaisesRegex(RuntimeError, "互斥"):
            next(p2.run_iter("", "", target_lang="Chinese"))
        it1.close()                                # 取消 → 锁释放
        it3 = p2.run_iter("", "", target_lang="Chinese")
        next(it3)                                  # 锁已释放, 可运行
        it3.close()


if __name__ == "__main__":
    unittest.main()
