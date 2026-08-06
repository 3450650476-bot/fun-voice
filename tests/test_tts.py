"""TTS 引擎回归测试 (fake 模型, 不加载真实权重)

覆盖: T2 参考长度可配 / T3 有效性校验 / T6 临时文件位置 / T8 生成参数 / batch_size 可配
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engines import tts as T


def _mk_wav(path, seconds, sr=16000, silent=False, amp=0.1):
    n = int(seconds * sr)
    data = np.zeros(n, np.float32) if silent else (np.random.randn(n) * amp).astype(np.float32)
    sf.write(path, data, sr)


class TestReferenceValidation(unittest.TestCase):
    """T3: 过短 / 近静音 / 无法读取均报错, 正常通过"""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.short = os.path.join(self.dir, "short.wav")
        self.silent = os.path.join(self.dir, "silent.wav")
        self.normal = os.path.join(self.dir, "normal.wav")
        self.bad = os.path.join(self.dir, "bad.txt")
        _mk_wav(self.short, 0.3)
        _mk_wav(self.silent, 1.0, silent=True)
        _mk_wav(self.normal, 1.0)
        with open(self.bad, "w") as f:
            f.write("not audio")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_short_rejected(self):
        with self.assertRaisesRegex(ValueError, "过短"):
            T.TTSEngine._validate_ref_audio(self.short)

    def test_silent_rejected(self):
        with self.assertRaisesRegex(ValueError, "近静音"):
            T.TTSEngine._validate_ref_audio(self.silent)

    def test_unreadable_rejected(self):
        with self.assertRaisesRegex(ValueError, "无法读取"):
            T.TTSEngine._validate_ref_audio(self.bad)

    def test_normal_ok(self):
        T.TTSEngine._validate_ref_audio(self.normal)    # 不抛


class TestClipRefAudio(unittest.TestCase):
    """T2/T6: 超长截取按 MAX_REF_SECONDS, 产物进系统 temp"""

    def test_long_clip_goes_to_tempdir(self):
        old = T.TTSEngine.MAX_REF_SECONDS
        T.TTSEngine.MAX_REF_SECONDS = 5
        try:
            with tempfile.TemporaryDirectory() as d:
                long = os.path.join(d, "long.wav")
                _mk_wav(long, 15)
                clip = T.TTSEngine._clip_ref_audio(long, 0.0)
                self.assertNotEqual(clip, long)
                self.assertIn("funvoice_ref_", os.path.basename(clip))
                self.assertEqual(os.path.dirname(os.path.abspath(clip)), tempfile.gettempdir())
        finally:
            T.TTSEngine.MAX_REF_SECONDS = old

    def test_short_clip_returns_original(self):
        with tempfile.TemporaryDirectory() as d:
            s = os.path.join(d, "s.wav")
            _mk_wav(s, 3)
            self.assertEqual(T.TTSEngine._clip_ref_audio(s, 0.0), s)


class TestGenKwargs(unittest.TestCase):
    """T8: 仅传已配置生成参数"""

    def tearDown(self):
        for k in ("VP_TTS_TEMPERATURE", "VP_TTS_TOP_P", "VP_TTS_TOP_K",
                  "VP_TTS_REPETITION_PENALTY"):
            os.environ.pop(k, None)

    def test_only_configured_passed(self):
        os.environ["VP_TTS_TEMPERATURE"] = "0.5"
        os.environ["VP_TTS_TOP_P"] = "0.9"
        e = T.TTSEngine(model_path="fake")
        self.assertEqual(e._gen_kwargs(), {"temperature": 0.5, "top_p": 0.9})

    def test_empty_when_unset(self):
        e = T.TTSEngine(model_path="fake")
        self.assertEqual(e._gen_kwargs(), {})

    def test_instance_args_override_env(self):
        os.environ["VP_TTS_TEMPERATURE"] = "0.5"
        e = T.TTSEngine(model_path="fake", temperature=0.2)
        self.assertEqual(e._gen_kwargs(), {"temperature": 0.2})


class TestBatchSize(unittest.TestCase):
    """BATCH_SIZE 可配: 参数 > env > 默认 12"""

    def tearDown(self):
        os.environ.pop("VP_TTS_BATCH_SIZE", None)

    def test_default_12(self):
        self.assertEqual(T.TTSEngine(model_path="fake").batch_size, 12)

    def test_explicit_wins(self):
        self.assertEqual(T.TTSEngine(model_path="fake", batch_size=24).batch_size, 24)

    def test_env_fallback(self):
        os.environ["VP_TTS_BATCH_SIZE"] = "18"
        self.assertEqual(T.TTSEngine(model_path="fake").batch_size, 18)


class TestCloneSynthesizeBatching(unittest.TestCase):
    """分批逻辑: 句数 > batch_size 时按批调用 generate_voice_clone"""

    def test_batches_by_batch_size(self):
        calls = []

        class M:
            def generate_voice_clone(self, text=None, **k):
                batch = text if isinstance(text, list) else [text]
                calls.append(batch)
                return [np.zeros(16000, np.float32) for _ in batch], 16000

        e = T.TTSEngine(model_path="fake", batch_size=12)
        e.model = M()
        e._prompt = object()
        wavs, sr = e.clone_synthesize([f"t{i}" for i in range(30)], "ref.wav")
        self.assertEqual([len(c) for c in calls], [12, 12, 6])
        self.assertEqual(len(wavs), 30)
        self.assertEqual(sr, 16000)


if __name__ == "__main__":
    unittest.main()
