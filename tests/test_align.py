"""时间轴对齐回归测试 (真实 librosa 变速, 不加载模型)

覆盖: 方案A 窗对齐兜底 (短句补静音/微超裁剪) + 方案B 间隙吸收回退 (末句漂移归零)
"""
import os
import shutil
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.pipeline as P
from app.engines.asr import Segment

SR = 16000
OUT_SR = 44100


def _noise(sec, sr=SR, amp=0.1):
    return (np.random.randn(int(sec * sr)) * amp).astype(np.float32)


class AlignTestCase(unittest.TestCase):
    def setUp(self):
        self.work = os.path.join("workspace", "_test_align")
        shutil.rmtree(self.work, ignore_errors=True)
        self.p = P.Pipeline(workspace=self.work)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _nonzero_span(self, wav, sr=OUT_SR):
        """返回波形非零部分的 (start, end) 秒 (容忍 5ms 底噪)"""
        amp = np.abs(wav)
        idx = np.where(amp > 1e-3)[0]
        if len(idx) == 0:
            return None
        return idx[0] / sr, idx[-1] / sr

    def test_short_clip_padded_to_window(self):
        # 方案A: 1s 配音填 4s 窗 —— 变速(0.556x→1.8s) + 尾部补静音(上限1.5s), 窗尾之后全静音
        segs = [Segment(0.0, 4.0, "x")]
        wavs = [(_noise(1.0), SR)]
        out, drift = self.p._align(segs, wavs, video_duration=4.0)
        self.assertEqual(out.shape, (2, (4 + 1) * OUT_SR))
        span = self._nonzero_span(out[0])
        self.assertIsNotNone(span)
        start, end = span
        self.assertAlmostEqual(start, 0.0, delta=0.05)
        self.assertTrue(1.5 <= end <= 2.0, f"变速后内容应约 1.8s, 实际 {end:.2f}")
        self.assertLess(drift, 0.05)                       # 短句不产生漂移
        # 补静音区 [3.4s, 4s] 必须全静音; 窗尾 4s 后也静音
        tail = np.max(np.abs(out[0, int(3.4 * OUT_SR):]))
        self.assertLess(tail, 1e-4)

    def test_overlong_clip_cropped(self):
        # 方案A: 超窗 ≤0.3s 裁剪到窗长 —— 3.6s 配音填 2s 窗 (变速极限 1.667x → 2.16s, 裁到 2s)
        segs = [Segment(0.0, 2.0, "x")]
        wavs = [(_noise(3.6), SR)]
        out, drift = self.p._align(segs, wavs, video_duration=2.0)
        span = self._nonzero_span(out[0])
        start, end = span
        self.assertAlmostEqual(start, 0.0, delta=0.05)
        self.assertTrue(1.9 <= end <= 2.1, f"应裁剪到 2s 窗, 实际 {end:.2f}")
        tail = np.max(np.abs(out[0, int(2.05 * OUT_SR):]))
        self.assertLess(tail, 1e-4)

    def test_drift_absorbed_by_gap(self):
        # 方案B: 前句短(留 2.7s 空隙) → 超窗句被推 1.6s → 后续句借空隙提前, 末句漂移归零
        # (无方案B时第3句被推至 9.6s, drift≈1.6s; 有方案B借空隙 → drift 归零)
        segs = [Segment(0.0, 6.0, "short"),     # 1s 配音: 变速 1.8s + 补静音 1.5s → 3.3s, 空隙 2.7s
                Segment(6.0, 8.0, "long"),      # 6s 配音填 2s 窗: 变速极限 1.667x → 3.6s, 超窗 1.6s
                Segment(8.0, 10.0, "ok")]       # 2s 配音填 2s 窗
        wavs = [(_noise(1.0), SR), (_noise(6.0), SR), (_noise(2.0), SR)]
        out, drift = self.p._align(segs, wavs, video_duration=10.0)
        self.assertLess(drift, 0.1, f"漂移应被空隙吸收, 实际 {drift:.2f}s")
        # 第 3 句写入 [8s, 10s] (借空隙回到原窗头, 覆盖超窗句尾部) — 窗尾 10s 后必须静音
        tail = np.max(np.abs(out[0, int(10.0 * OUT_SR):]))
        self.assertLess(tail, 1e-4)

    def test_natural_short_centered(self):
        # natural 模式: 3s 配音填 6s 窗 —— 不变速, 中点对齐 → 内容 [1.5, 4.5], 前后对称静音
        segs = [Segment(0.0, 6.0, "x")]
        wavs = [(_noise(3.0), SR)]
        out, drift = self.p._align(segs, wavs, video_duration=6.0, mode="natural")
        span = self._nonzero_span(out[0])
        self.assertIsNotNone(span)
        start, end = span
        self.assertAlmostEqual(start, 1.5, delta=0.05, msg=f"短句应中点对齐到 1.5s, 实际 {start:.2f}")
        self.assertAlmostEqual(end, 4.5, delta=0.05, msg=f"内容应自然语速 3s, 实际 {end - start:.2f}")
        self.assertLess(drift, 0.05)
        # 前后空隙必须对称静音
        head = np.max(np.abs(out[0, :int(1.4 * OUT_SR)]))
        tail = np.max(np.abs(out[0, int(4.6 * OUT_SR):]))
        self.assertLess(head, 1e-4)
        self.assertLess(tail, 1e-4)

    def test_natural_long_compressed_same_as_stretch(self):
        # 长句(超窗)在两种模式下行为一致: 压缩 + 裁剪到窗长
        segs = [Segment(0.0, 2.0, "x")]
        wavs = [(_noise(3.6), SR)]
        out_n, drift_n = self.p._align(segs, wavs, video_duration=2.0, mode="natural")
        out_s, drift_s = self.p._align(segs, wavs, video_duration=2.0, mode="stretch")
        np.testing.assert_array_equal(out_n, out_s)
        self.assertLess(drift_n, 0.05)
        self.assertLess(drift_s, 0.05)

    def test_exact_window_alignment(self):
        # 常规场景: 配音恰等于窗长, 无变速无补静音, 拼接无漂移无空隙
        segs = [Segment(0.0, 2.0, "a"), Segment(2.0, 4.0, "b")]
        wavs = [(_noise(2.0), SR), (_noise(2.0), SR)]
        out, drift = self.p._align(segs, wavs, video_duration=4.0)
        self.assertLess(drift, 0.05)
        # 第 2 句应从 2s 处开始
        gap = np.max(np.abs(out[0, int(1.9 * OUT_SR):int(2.1 * OUT_SR)]))
        self.assertGreater(gap, 0.01)   # 2s 附近有语音(第2句起点), 非静音空隙


if __name__ == "__main__":
    unittest.main()
