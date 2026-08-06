"""管道级回归测试 (fake 引擎, 全链路秒级, 不加载真实模型)

覆盖: yield 序列 (T7) / B2 异常与取消释放 / batch_size 透传 / 自动模式移除 / B5 rerun 参数沿用
"""
import os
import shutil
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.pipeline as P
from app.engines.asr import Segment


class FakeSep:
    def separate(self, wave, sr):
        return np.zeros((2, 44100), np.float32), np.zeros((2, 44100), np.float32)


class FakeASR:
    def transcribe(self, *a, **k):
        return [Segment(0.0, 2.0, "hello")]

    def release(self):
        pass


class BoomASR(FakeASR):
    def transcribe(self, *a, **k):
        raise RuntimeError("boom-asr")


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


def _align_stub(self, segs, wavs, dur, smin, smax, mode="stretch"):
    return np.zeros((2, int(dur * 44100) + 44100), np.float32), 0.0


class PipelineTestCase(unittest.TestCase):
    EXP_SEQ = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7]     # 每阶段: 开始提示(含预估)+完成; 阶段5 四条(开始/加载/批量/完成)

    def setUp(self):
        self._patchers = [
            mock.patch.object(P, "extract_audio", lambda *a, **k: None),
            mock.patch.object(P, "probe_video_info", lambda p: {"duration": 10.0}),
            mock.patch.object(P, "read_wav", lambda p: (np.zeros((2, 44100), np.float32), 44100)),
            mock.patch.object(P, "write_wav", lambda *a, **k: None),
            mock.patch.object(P, "match_loudness", lambda a, b: np.asarray(b, np.float32)),
            mock.patch.object(P, "mix_to_video", lambda *a, **k: "out.mp4"),
            mock.patch.object(P, "get_separator", lambda *a, **k: FakeSep()),
            mock.patch.object(P, "get_translator", lambda **k: FakeTrans()),
            mock.patch.object(P, "ASREngine", FakeASR),
            mock.patch.object(P, "TTSEngine", FakeTTS),
            mock.patch.object(P.Pipeline, "_align", _align_stub),
        ]
        for p in self._patchers:
            p.start()
        self.work = os.path.join("workspace", "_test_pipeline")
        shutil.rmtree(self.work, ignore_errors=True)
        self.p = P.Pipeline(workspace=self.work)

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        shutil.rmtree(self.work, ignore_errors=True)
        if os.path.exists(P.RunLock.LOCK_PATH):   # 防御: 断言失败等异常路径保证锁不残留
            os.remove(P.RunLock.LOCK_PATH)

    # ---------- yield 序列与消息 ----------

    def test_full_sequence(self):
        items = list(self.p.run_iter("v.mp4", ref_audio="ref.wav", target_lang="Chinese"))
        self.assertEqual([s for s, _, _ in items], self.EXP_SEQ)
        s5_msgs = [m for s, _, m in items if s == 5]
        self.assertEqual(len(s5_msgs), 4)
        self.assertIn("正在克隆配音", s5_msgs[0])
        self.assertIn("预计", s5_msgs[0])          # 方案 A: 预估在阶段开始时出现
        self.assertIn("正在加载 TTS 模型", s5_msgs[1])
        self.assertIn("克隆配音完成", s5_msgs[3])
        self.assertIn("｜实际", s5_msgs[3])         # 完成消息只留实际用时
        self.assertNotIn("预估", s5_msgs[3])

    # ---------- 自动模式移除 ----------

    def test_no_ref_audio_raises(self):
        with self.assertRaisesRegex(ValueError, "配音音色"):
            list(self.p.run_iter("v.mp4", ref_audio=None, target_lang="Chinese"))

    # ---------- 完成至翻译 (stop_after=4) ----------

    def test_stop_after_translate(self):
        items = list(self.p.run_iter("v.mp4", ref_audio="ref.wav",
                                     target_lang="Chinese", stop_after=4))
        seq = [s for s, _, _ in items]
        self.assertEqual(seq, [1, 1, 2, 2, 3, 3, 4, 4, 4])     # 阶段1-4 各开始+完成 + 收尾
        self.assertTrue(items[-1][2].startswith("✅ 已停止于「翻译完成」"))
        res = items[-1][1]
        self.assertTrue(res.zh_lines)          # 译文已生成
        self.assertFalse(res.dubbed_audio)     # 未配音 (dataclass 默认空串)
        self.assertFalse(res.output_video)     # 未混流
        self.assertTrue(os.path.isfile(os.path.join(self.work, "state.json")))

    def test_stop_after_translate_then_resume_completes(self):
        list(self.p.run_iter("v.mp4", ref_audio="ref.wav",
                             target_lang="Chinese", stop_after=4))
        # 续跑: 阶段3/4 跳过, 从阶段5 继续到完成
        p2 = P.Pipeline(workspace=self.work)
        items2 = list(p2.run_iter("", "", target_lang="Chinese"))
        joined = "\n".join(m for _, _, m in items2)
        self.assertIn("跳过识别", joined)
        self.assertIn("跳过翻译", joined)
        self.assertIn("克隆配音完成", joined)
        self.assertTrue(items2[-1][2].startswith("✅ 全部完成"))
        res2 = items2[-1][1]
        self.assertTrue(res2.dubbed_audio)     # 已配音 (mock 不落盘, 仅验证路径已赋值)
        self.assertTrue(res2.output_video)

    # ---------- B2: 异常/取消时资源释放 ----------

    def test_asr_exception_releases_model(self):
        released = []

        class BoomASR2(FakeASR):
            def transcribe(self, *a, **k):
                raise RuntimeError("boom-asr")

            def release(self):
                released.append(True)

        with mock.patch.object(P, "ASREngine", BoomASR2):
            with self.assertRaisesRegex(RuntimeError, "boom-asr"):
                list(self.p.run_iter("v.mp4", ref_audio="ref.wav", target_lang="Chinese"))
        self.assertEqual(released, [True])

    def test_cancel_releases_tts(self):
        released = []

        class TTSTracking(FakeTTS):
            def release(self):
                released.append(True)

        with mock.patch.object(P, "TTSEngine", TTSTracking):
            it = self.p.run_iter("v.mp4", ref_audio="ref.wav", target_lang="Chinese")
            stage = None
            for _ in range(11):       # 消费到 [5/7] TTS 批量完成 yield (新序列第 11 个)
                stage, _, _ = next(it)
            self.assertEqual(stage, 5)
            it.close()                # GeneratorExit -> finally -> tts.release() + lock.release()
        self.assertEqual(released, [True])

    # ---------- batch_size 透传 ----------

    def test_batch_size_passthrough(self):
        seen = {}

        class TTSTracking(FakeTTS):
            def __init__(self, **kw):
                super().__init__(**kw)
                seen["batch_size"] = kw.get("batch_size")

        with mock.patch.object(P, "TTSEngine", TTSTracking):
            list(self.p.run_iter("v.mp4", ref_audio="ref.wav",
                                 target_lang="Chinese", batch_size=20))
        self.assertEqual(seen["batch_size"], 20)

    def test_asr_config_passthrough(self):
        seen = {}

        class ASRSpy(FakeASR):
            def __init__(self, **kw):
                seen["cfg"] = kw

        with mock.patch.object(P, "ASREngine", ASRSpy):
            list(self.p.run_iter("v.mp4", ref_audio="ref.wav", target_lang="Chinese",
                                 asr_config={"hotwords": "华盛顿", "initial_prompt": "新闻"}))
        self.assertEqual(seen["cfg"], {"hotwords": "华盛顿", "initial_prompt": "新闻"})

    # ---------- B5: rerun_segment 参数沿用 ----------

    def _make_res(self):
        res = P.PipelineResult(workspace=self.work, video="v.mp4",
                               volume_gain=1.5, quality="balanced", stretch=(0.5, 2.0))
        res.source_audio = "src.wav"
        res.dubbed_audio = "dub.wav"
        res.asr_segments = [(0.0, 2.0, "hello")]
        res.zh_lines = ["译:你好"]
        return res

    def test_rerun_uses_res_stretch_and_quality(self):
        stretch_seen, quality_seen = [], []

        def align_spy(self, segs, wavs, dur, smin, smax, mode="stretch"):
            stretch_seen.append((smin, smax))
            return np.zeros((2, 10 * 44100), np.float32), 0.0

        def mix_spy(*a, **k):
            quality_seen.append(k.get("quality"))
            return "out.mp4"

        with mock.patch.object(P.Pipeline, "_align", align_spy), \
             mock.patch.object(P, "mix_to_video", mix_spy):
            new_res, msg = self.p.rerun_segment(self._make_res(), 0, "新译文",
                                                ref_audio="ref.wav")
        self.assertIsNotNone(new_res)
        self.assertEqual(stretch_seen, [(0.5, 2.0)])
        self.assertEqual(quality_seen, ["balanced"])

    def test_rerun_explicit_stretch_wins(self):
        stretch_seen = []

        def align_spy(self, segs, wavs, dur, smin, smax, mode="stretch"):
            stretch_seen.append((smin, smax))
            return np.zeros((2, 10 * 44100), np.float32), 0.0

        with mock.patch.object(P.Pipeline, "_align", align_spy):
            self.p.rerun_segment(self._make_res(), 0, "新译文2", ref_audio="ref.wav",
                                 stretch=(0.6, 1.8))
        self.assertEqual(stretch_seen, [(0.6, 1.8)])

    def test_rerun_no_ref_raises(self):
        with self.assertRaisesRegex(ValueError, "配音音色"):
            self.p.rerun_segment(self._make_res(), 0, "新译文", ref_audio=None)


if __name__ == "__main__":
    unittest.main()
