"""音频工具回归测试 (mock ffmpeg 子进程, 不依赖真实 ffmpeg)

覆盖: A1 路径 env 覆盖 / A5 probe lru_cache 去重 / A2-A3 extract_audio 错误分支
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.audio as A

ENV_KEYS = ("VP_MODELS", "FFMPEG_PATH")


class TestPaths(unittest.TestCase):
    """A1: 默认相对定位 + env 覆盖 (reload 隔离)"""

    def setUp(self):
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        importlib.reload(A)

    def test_default_relative_paths(self):
        importlib.reload(A)
        exp = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(A.__file__)))),
            "models", "ffmpeg", "bin", "ffmpeg.exe")
        self.assertEqual(A.FFMPEG_PATH, exp)

    def test_vp_models_override(self):
        tmp = tempfile.mkdtemp()
        try:
            os.environ["VP_MODELS"] = tmp
            importlib.reload(A)
            self.assertEqual(A.FFMPEG_PATH, os.path.join(tmp, "ffmpeg", "bin", "ffmpeg.exe"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ffmpeg_path_override(self):
        os.environ["FFMPEG_PATH"] = r"C:\fake\ffmpeg.exe"
        importlib.reload(A)
        self.assertEqual(A.FFMPEG_PATH, r"C:\fake\ffmpeg.exe")
        self.assertEqual(A.FFPROBE_PATH, r"C:\fake\ffprobe.exe")


class TestProbeCache(unittest.TestCase):
    """A5: probe_video_info / probe_duration lru_cache 去重"""

    def setUp(self):
        A.probe_video_info.cache_clear()
        A.probe_duration.cache_clear()

    def test_cache_hit_no_extra_subprocess(self):
        payload = json.dumps({
            "format": {"duration": "10.0"},
            "streams": [{"codec_type": "video", "codec_name": "h264"}]})
        fake = mock.Mock(returncode=0, stdout=payload, stderr="")
        with mock.patch.object(A.subprocess, "run", return_value=fake) as mr:
            info = A.probe_video_info("v.mp4")
            self.assertEqual(info["duration"], 10.0)
            self.assertEqual(mr.call_count, 1)
            A.probe_video_info("v.mp4")     # probe_video_info 缓存命中
            self.assertEqual(mr.call_count, 1)
        # probe_duration 是独立 lru_cache (ffprobe 输出纯数字)
        A.probe_duration.cache_clear()
        fake2 = mock.Mock(returncode=0, stdout="10.0\n", stderr="")
        with mock.patch.object(A.subprocess, "run", return_value=fake2) as mr:
            self.assertEqual(A.probe_duration("v.mp4"), 10.0)
            self.assertEqual(mr.call_count, 1)
            A.probe_duration("v.mp4")       # probe_duration 缓存命中
            self.assertEqual(mr.call_count, 1)

    def test_new_path_probes_once(self):
        payload = json.dumps({"format": {}, "streams": []})
        fake = mock.Mock(returncode=0, stdout=payload, stderr="")
        with mock.patch.object(A.subprocess, "run", return_value=fake) as mr:
            A.probe_video_info("a.mp4")
            A.probe_video_info("b.mp4")
            self.assertEqual(mr.call_count, 2)


class TestExtractAudio(unittest.TestCase):
    """A2/A3: 无音轨 / 流越界 / ffmpeg 失败 / 成功"""

    def test_no_audio_stream_raises(self):
        with mock.patch.object(A, "probe_audio_streams", return_value=0):
            with self.assertRaisesRegex(ValueError, "没有音轨"):
                A.extract_audio("v.mp4", "out.wav")

    def test_stream_index_out_of_range(self):
        with mock.patch.object(A, "probe_audio_streams", return_value=2):
            with self.assertRaisesRegex(ValueError, "只有 2 条音轨"):
                A.extract_audio("v.mp4", "out.wav", audio_stream=2)

    def test_ffmpeg_failure_raises(self):
        with mock.patch.object(A, "probe_audio_streams", return_value=1):
            with mock.patch.object(A.subprocess, "run",
                                   return_value=mock.Mock(returncode=1, stderr="boom")):
                with self.assertRaises(RuntimeError):
                    A.extract_audio("v.mp4", "out.wav")

    def test_success_returns_path(self):
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "out.wav")
            with open(out, "wb") as f:
                f.write(b"x" * 100)      # 模拟 ffmpeg 产物 (>44 字节)
            with mock.patch.object(A, "probe_audio_streams", return_value=1):
                with mock.patch.object(A.subprocess, "run",
                                       return_value=mock.Mock(returncode=0)):
                    self.assertEqual(A.extract_audio("v.mp4", out), out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestMixToVideo(unittest.TestCase):
    """混流出片: faststart / -t 时长 / copy 失败回退重编码 / probe 失败退化"""

    def setUp(self):
        A.probe_video_info.cache_clear()

    def _cmd(self, mr):
        return mr.call_args.args[0]

    def test_copy_uses_faststart_and_t(self):
        with mock.patch.object(A, "probe_video_info", return_value={"duration": 10.0}):
            with mock.patch.object(A.subprocess, "run",
                                   return_value=mock.Mock(returncode=0)) as mr:
                r = A.mix_to_video("v.mp4", "dub.wav", "out.mp4", quality="copy")
        self.assertEqual(r, "out.mp4")
        cmd = self._cmd(mr)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")   # 无损直通
        self.assertIn("-movflags", cmd) and self.assertIn("+faststart", cmd)
        self.assertIn("-t", cmd)
        self.assertIn("10.000", cmd)
        self.assertNotIn("-shortest", cmd)

    def test_copy_failure_falls_back_to_reencode(self):
        with mock.patch.object(A, "probe_video_info", return_value={"duration": 10.0}):
            with mock.patch.object(A.subprocess, "run",
                                   side_effect=[mock.Mock(returncode=1, stderr="boom"),
                                                mock.Mock(returncode=0)]) as mr:
                r = A.mix_to_video("v.mp4", "dub.wav", "out.mp4", quality="copy")
        self.assertEqual(r, "out.mp4")
        self.assertEqual(mr.call_count, 2)
        cmd1 = mr.call_args_list[0].args[0]
        cmd2 = mr.call_args_list[1].args[0]
        self.assertEqual(cmd1[cmd1.index("-c:v") + 1], "copy")
        self.assertEqual(cmd2[cmd2.index("-c:v") + 1], "libx264")
        self.assertEqual(cmd2[cmd2.index("-crf") + 1], "23")

    def test_small_quality_uses_crf28(self):
        with mock.patch.object(A, "probe_video_info", return_value={"duration": 10.0}):
            with mock.patch.object(A.subprocess, "run",
                                   return_value=mock.Mock(returncode=0)) as mr:
                A.mix_to_video("v.mp4", "dub.wav", "out.mp4", quality="small")
        cmd = self._cmd(mr)
        self.assertEqual(cmd[cmd.index("-crf") + 1], "28")

    def test_probe_failure_falls_back_to_shortest(self):
        with mock.patch.object(A, "probe_video_info", return_value=None):
            with mock.patch.object(A.subprocess, "run",
                                   return_value=mock.Mock(returncode=0)) as mr:
                A.mix_to_video("v.mp4", "dub.wav", "out.mp4", quality="copy")
        cmd = self._cmd(mr)
        self.assertIn("-shortest", cmd)
        self.assertNotIn("-t", cmd)
        self.assertIn("+faststart", cmd)

    def test_background_wav_adds_amix(self):
        with mock.patch.object(A, "probe_video_info", return_value={"duration": 10.0}):
            with mock.patch.object(A.subprocess, "run",
                                   return_value=mock.Mock(returncode=0)) as mr:
                A.mix_to_video("v.mp4", "dub.wav", "out.mp4", quality="copy",
                               background_wav="inst.wav")
        cmd = self._cmd(mr)
        self.assertEqual(cmd.count("-i"), 3)                  # 视频 + 配音 + 背景
        self.assertIn("inst.wav", cmd)
        self.assertIn("-filter_complex", cmd)
        af = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("amix=inputs=2:duration=first:normalize=0", af)
        self.assertIn("[aout]", cmd)
        self.assertNotIn("1:a:0", cmd)   # 背景混合时不直接 map 原配音

    def test_no_background_keeps_single_audio(self):
        with mock.patch.object(A, "probe_video_info", return_value={"duration": 10.0}):
            with mock.patch.object(A.subprocess, "run",
                                   return_value=mock.Mock(returncode=0)) as mr:
                A.mix_to_video("v.mp4", "dub.wav", "out.mp4", quality="copy")
        cmd = self._cmd(mr)
        self.assertEqual(cmd.count("-i"), 2)                  # 视频 + 配音
        self.assertNotIn("-filter_complex", cmd)
        maps = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-map"]
        self.assertEqual(maps, ["0:v:0", "1:a:0"])


if __name__ == "__main__":
    unittest.main()
