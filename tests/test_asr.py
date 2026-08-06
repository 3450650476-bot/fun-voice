"""ASR 引擎回归测试 (mock faster_whisper, 不加载真实模型)

覆盖: hotwords/initial_prompt 透传 (构造与 transcribe 级) / env 兜底 (VP_ASR_*)
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engines.asr import ASREngine

ENV = ("VP_ASR_HOTWORDS", "VP_ASR_PROMPT", "VP_WHISPER_COMPUTE", "VP_ASR_LANG")


class FakeSeg:
    def __init__(self, s, e, t):
        self.start, self.end, self.text, self.words = s, e, t, None


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kw):
        self.calls.append(kw)
        return iter([FakeSeg(0.0, 1.0, "hello")]), None


class TestASRConfig(unittest.TestCase):
    def setUp(self):
        for k in ENV:
            os.environ.pop(k, None)
        self.model = FakeModel()
        self.patcher = mock.patch("faster_whisper.WhisperModel",
                                  lambda *a, **k: self.model)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        for k in ENV:
            os.environ.pop(k, None)

    def test_constructor_config_passed_to_transcribe(self):
        eng = ASREngine(initial_prompt="新闻播报", hotwords="华盛顿,美联储")
        segs = eng.transcribe("x.wav")
        kw = self.model.calls[0]
        self.assertEqual(kw["hotwords"], "华盛顿,美联储")
        self.assertEqual(kw["initial_prompt"], "新闻播报")
        self.assertEqual(kw["beam_size"], 5)
        self.assertTrue(kw["vad_filter"])
        self.assertEqual(len(segs), 1)

    def test_env_fallback(self):
        os.environ["VP_ASR_HOTWORDS"] = "Biden,WHO"
        os.environ["VP_ASR_PROMPT"] = "新闻"
        eng = ASREngine()
        eng.transcribe("x.wav")
        kw = self.model.calls[0]
        self.assertEqual(kw["hotwords"], "Biden,WHO")
        self.assertEqual(kw["initial_prompt"], "新闻")

    def test_transcribe_arg_overrides_constructor(self):
        eng = ASREngine(hotwords="默认词")
        eng.transcribe("x.wav", hotwords="临时词")
        self.assertEqual(self.model.calls[0]["hotwords"], "临时词")

    def test_none_config_not_passed(self):
        eng = ASREngine()
        eng.transcribe("x.wav")
        kw = self.model.calls[0]
        self.assertIsNone(kw["hotwords"])
        self.assertIsNone(kw["initial_prompt"])


if __name__ == "__main__":
    unittest.main()
