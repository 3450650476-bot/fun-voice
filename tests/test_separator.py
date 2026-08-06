"""人声分离引擎回归测试 (fake 模型, 不加载真实权重)

覆盖: B1 pad 边界 / num_overlap 重叠平均 / min_mean_abs 静音跳过 / 空输入保护
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engines import separator as sep_mod


class FakeMDX:
    def __init__(self):
        self.calls = 0

    def to(self, d):
        return self

    def eval(self):
        return self

    def __call__(self, chunk):
        self.calls += 1
        return torch.full((1, 2, 2, chunk.shape[-1]), 0.5, dtype=torch.float32)


class TestMDX23C(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeMDX()
        cls.patcher = mock.patch.object(sep_mod, "load_mdx23c", lambda *a, **k: cls.fake)
        cls.patcher.start()
        cls.s = sep_mod.MDX23Separator(device="cpu")
        cls.gen = cls.s.chunk_size - 2 * cls.s.trim

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def tearDown(self):
        self.fake.calls = 0
        self.s.cfg = dict(sep_mod.MDX23C_CFG)

    def test_empty_input_no_inference(self):
        v, i = self.s.separate(np.zeros((2, 0), np.float32), 44100)
        self.assertEqual(v.shape, (2, 0))
        self.assertEqual(self.fake.calls, 0)

    def test_pad_boundary_integer_multiple(self):
        # B1 回归: n 恰为 gen 整数倍时只跑 1 块 (修复前多算一整块 GPU 推理)
        wave = np.random.randn(2, self.gen).astype(np.float32)
        v, i = self.s.separate(wave, 44100)
        self.assertEqual(self.fake.calls, 1)
        self.assertEqual(v.shape, (2, self.gen))

    def test_pad_boundary_non_multiple(self):
        self.s.cfg = {**sep_mod.MDX23C_CFG, "num_overlap": 1}
        wave = np.random.randn(2, self.gen + 1).astype(np.float32)
        v, i = self.s.separate(wave, 44100)
        self.assertEqual(self.fake.calls, 2)
        self.assertEqual(v.shape, (2, self.gen + 1))

    def test_num_overlap_4_window_average(self):
        self.s.cfg = {**sep_mod.MDX23C_CFG, "num_overlap": 4}
        wave = np.random.randn(2, 2 * self.gen).astype(np.float32)
        v, i = self.s.separate(wave, 44100)
        self.assertEqual(self.fake.calls, 5)
        np.testing.assert_allclose(v, 0.5, atol=1e-6)

    def test_silence_skip(self):
        self.s.cfg = {**sep_mod.MDX23C_CFG, "num_overlap": 4, "min_mean_abs": 0.001}
        v, i = self.s.separate(np.zeros((2, 2 * self.gen), np.float32), 44100)
        self.assertEqual(self.fake.calls, 0)
        np.testing.assert_array_equal(v, 0)


class FakeSess:
    def __init__(self):
        self.calls = 0

    def run(self, *a, **k):
        self.calls += 1
        return [np.zeros((1, 4, 3072, 256), np.float32)]


ONNX_MODEL = "UVR-MDX-NET-Inst_HQ_3"
ONNX_PATH = os.path.join(sep_mod.MDX_MODELS_DIR, f"{ONNX_MODEL}.onnx")


@unittest.skipUnless(os.path.isfile(ONNX_PATH), f"缺少 onnx 模型: {ONNX_PATH}")
class TestMDXOnnx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime
        cls.sess = FakeSess()
        cls.patcher = mock.patch.object(onnxruntime, "InferenceSession",
                                        lambda *a, **k: cls.sess)
        cls.patcher.start()
        cls.s = sep_mod.MDXOnnxSeparator(ONNX_MODEL)
        cls.gen = cls.s.chunk_size - 2 * cls.s.trim
        cls.warmup = cls.sess.calls

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def tearDown(self):
        self.sess.calls = self.warmup
        self.s.cfg = dict(sep_mod.ONNX_MODELS[ONNX_MODEL])

    def test_empty_input(self):
        v, i = self.s.separate(np.zeros((2, 0), np.float32), 44100)
        self.assertEqual(v.shape, (2, 0))
        self.assertEqual(self.sess.calls, self.warmup)

    def test_pad_boundary_integer_multiple(self):
        wave = np.random.randn(2, self.gen).astype(np.float32)
        v, i = self.s.separate(wave, 44100)
        self.assertEqual(self.sess.calls, self.warmup + 1)
        self.assertEqual(v.shape, (2, self.gen))

    def test_silence_skip(self):
        self.s.cfg = {**sep_mod.ONNX_MODELS[ONNX_MODEL], "min_mean_abs": 0.001}
        v, i = self.s.separate(np.zeros((2, 2 * self.gen), np.float32), 44100)
        self.assertEqual(self.sess.calls, self.warmup)
        np.testing.assert_array_equal(v, 0)


if __name__ == "__main__":
    unittest.main()
