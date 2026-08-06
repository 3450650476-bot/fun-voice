"""翻译引擎回归测试 (mock openai / transformers, 无网络)

覆盖: T1 配置优先级 / temperature 容错 / B6 降级链与行数对齐 / 跨批上下文
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engines import translator as T

ENV = ("VP_TRANSLATE_API_KEY", "VP_TRANSLATE_BASE_URL", "VP_TRANSLATE_MODEL",
       "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL")


class FakeResp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class FakeCreate:
    def __init__(self, fail_temperature_once=False):
        self.calls = 0
        self.fail_temperature_once = fail_temperature_once

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail_temperature_once and self.calls == 1:
            raise RuntimeError("temperature is not supported for this model")
        user = kwargs["messages"][1]["content"]
        body = user.split(":\n")[-1]
        return FakeResp("\n".join(f"译:{ln}" for ln in body.split("\n")))


class FakeCompletions:
    def __init__(self, fc):
        self.create = fc


class FakeChat:
    def __init__(self, fc):
        self.completions = FakeCompletions(fc)


class FakeClient:
    def __init__(self, *a, **k):
        self.chat = FakeChat(FakeCreate())


class TestConfigPriority(unittest.TestCase):
    """T1: 实例 > VP_TRANSLATE_* > DEEPSEEK_* > 默认"""

    def tearDown(self):
        for k in ENV:
            os.environ.pop(k, None)

    def test_instance_overrides_env(self):
        t = T.OpenAICompatTranslator(api_key="k1", base_url="https://x.com/v1", model="qwen-max")
        self.assertEqual((t.api_key, t.base_url, t.model), ("k1", "https://x.com/v1", "qwen-max"))

    def test_env_fallback_priority(self):
        os.environ["DEEPSEEK_API_KEY"] = "dk"
        os.environ["DEEPSEEK_MODEL"] = "deepseek-chat"
        os.environ["VP_TRANSLATE_MODEL"] = "glm-4-flash"
        t = T.OpenAICompatTranslator()
        self.assertEqual(t.api_key, "dk")
        self.assertEqual(t.model, "glm-4-flash")
        self.assertEqual(t.base_url, "https://api.deepseek.com")


class TestTemperatureFallback(unittest.TestCase):
    """T1: reasoner 类模型不支持 temperature -> 去掉后重试"""

    def tearDown(self):
        for k in ENV:
            os.environ.pop(k, None)

    def test_temperature_retry_without_param(self):
        os.environ["DEEPSEEK_API_KEY"] = "dk"
        fc = FakeCreate(fail_temperature_once=True)
        calls = []

        def spy(**kwargs):
            calls.append(dict(kwargs))
            return fc(**kwargs)

        client = mock.Mock()
        client.chat.completions.create = spy
        with mock.patch("openai.OpenAI", return_value=client):
            t = T.OpenAICompatTranslator()
            t._client = client
            r = t.translate_text("hello")
        self.assertEqual(r, "译:hello")
        self.assertIsNone(t.temperature)
        self.assertIn("temperature", calls[0])
        self.assertNotIn("temperature", calls[1])


class FakePipe:
    def __call__(self, text):
        ls = text if isinstance(text, list) else [text]
        return [{"translation_text": "译:" + t} for t in ls]


class TestFallbackChain(unittest.TestCase):
    """B6/T3: 无 key 自动降级本地兜底; 兜底也失败保留原文"""

    def setUp(self):
        for k in ENV:
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ENV:
            os.environ.pop(k, None)

    def test_no_key_falls_back_to_local(self):
        with mock.patch.object(T.LocalFallbackTranslator, "_ensure",
                               lambda self: setattr(self, "_pipe", FakePipe())):
            tr = T.get_translator()
            r = tr.translate_lines(["hello", "world"], batch_size=10, max_retries=1)
        self.assertEqual(r, ["译:hello", "译:world"])

    def test_fallback_failure_keeps_original(self):
        class BoomPipe:
            def __call__(self, text):
                raise RuntimeError("opus offline")
        with mock.patch.object(T.LocalFallbackTranslator, "_ensure",
                               lambda self: setattr(self, "_pipe", BoomPipe())):
            tr = T.get_translator()
            r = tr.translate_lines(["hello", "world"], batch_size=10, max_retries=1)
        self.assertEqual(r, ["hello", "world"])

    def test_partial_batch_failure_falls_back_per_line(self):
        def fake2(text, **k):
            ls = text.split("\n")
            if len(ls) > 1:
                return ls[0]               # 批量只回 1 行 -> 触发逐条
            if ls[0].startswith("bad"):
                raise RuntimeError("net error")
            return "译:" + ls[0]
        with mock.patch.object(T.LocalFallbackTranslator, "_ensure",
                               lambda self: setattr(self, "_pipe", FakePipe())):
            t = T.OpenAICompatTranslator(api_key="k")
            t.translate_text = fake2
            r = t.translate_lines(["a", "bad1", "c"], batch_size=10, max_retries=2)
        self.assertEqual(r, ["译:a", "译:bad1", "译:c"])   # bad1 走本地兜底


class TestCrossBatchContext(unittest.TestCase):
    """A/B: 第二批携带上批末尾作参考上下文"""

    def setUp(self):
        os.environ["DEEPSEEK_API_KEY"] = "dk"

    def tearDown(self):
        for k in ENV:
            os.environ.pop(k, None)

    def test_second_batch_has_context(self):
        calls = []
        fc = FakeCreate()

        def spy(**kwargs):
            calls.append(dict(kwargs))
            return fc(**kwargs)

        client = mock.Mock()
        client.chat.completions.create = spy
        with mock.patch("openai.OpenAI", return_value=client):
            tr = T.get_translator()
            tr._client = client
            lines = [f"L{i}" for i in range(1, 16)]     # 15 行, batch=10 -> 两批
            r = tr.translate_lines(lines, batch_size=10, max_retries=1)
        self.assertEqual(len(r), 15)
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotIn("参考上下文", calls[0]["messages"][1]["content"])
        self.assertIn("参考上下文", calls[1]["messages"][1]["content"])
        self.assertIn("L10", calls[1]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
