"""翻译引擎: 通用 OpenAI 兼容 API (主) + 本地 OPUS-MT 离线兜底

主引擎: OpenAICompatTranslator — 兼容 DeepSeek/通义/Kimi/GLM/SiliconFlow/Ollama 等
    配置: base_url + api_key + model (构造参数 > VP_TRANSLATE_* env > DEEPSEEK_* env > 默认)
兜底: LocalFallbackTranslator — Helsinki-NLP/opus-mt-en-zh (transformers, 免 key 离线, 仅 en→zh)
优先级链: 主引擎 → 本地兜底 → 保留原文 (translate_lines 内自动降级, 不中断任务)

统一接口: translate_text(text, target_lang) / translate_lines(lines, ...) / translate_subtitle(...)
"""
from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# 向后兼容的 DeepSeek 默认 (仍可用 DEEPSEEK_* 配置)
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 语言名 -> 翻译提示词
LANG_PROMPT = {
    "Chinese": "翻译为简体中文",
    "English": "translate into English",
    "Japanese": "日本語に翻訳してください",
    "Korean": "한국어로 번역해 주세요",
}


def _env(*names, default=None):
    """依次取 env, 返回第一个非空值或 default"""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def deepseek_available() -> bool:
    """是否有可用的主引擎 key (兼容旧 DEEPSEEK_* 与新 VP_TRANSLATE_*)"""
    return bool(_env("VP_TRANSLATE_API_KEY", "DEEPSEEK_API_KEY"))


class OpenAICompatTranslator:
    """通用 OpenAI 兼容翻译器 (DeepSeek/通义/Kimi/GLM/SiliconFlow/Ollama 等).

    配置优先级: 构造参数 > VP_TRANSLATE_BASE_URL/API_KEY/MODEL > DEEPSEEK_* > 默认值.
    temperature=None 时请求不带该参数 (兼容 reasoner 类不接受 temperature 的模型).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, temperature: Optional[float] = 0.3,
                 max_tokens: Optional[int] = 4096,
                 fallback: Optional["LocalFallbackTranslator"] = None):
        self.api_key = api_key or _env("VP_TRANSLATE_API_KEY", "DEEPSEEK_API_KEY")
        self.base_url = base_url or _env("VP_TRANSLATE_BASE_URL", "DEEPSEEK_BASE_URL",
                                         default=DEEPSEEK_BASE_URL)
        self.model = model or _env("VP_TRANSLATE_MODEL", "DEEPSEEK_MODEL", default=DEEPSEEK_MODEL)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback = fallback if fallback is not None else LocalFallbackTranslator()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key or "missing-key",
                                  base_url=self.base_url)
        return self._client

    def translate_text(self, text: str, target_lang: str = "Chinese",
                       source_lang: Optional[str] = None,
                       max_retries: int = 3, context: Optional[str] = None) -> str:
        """翻译整段文本 (OpenAI 兼容 chat 接口)
        context: 可选参考上下文 (仅帮助理解, 不参与输出行数) — 解决碎片句跨批上下文丢失"""
        if not self.api_key:
            raise RuntimeError("未配置翻译 API key (VP_TRANSLATE_API_KEY / DEEPSEEK_API_KEY)")
        instr = LANG_PROMPT.get(target_lang, f"translate into {target_lang}")
        system = ("你是专业字幕翻译。只输出翻译结果，不要解释，不要添加任何额外内容。"
                  "保持每行的句数与原文一致，逐行对应输出。"
                  "相邻行可能属于同一句话(尤其是不完整短语)，翻译时必须结合上下文理解整句语义，"
                  "不要机械地逐字翻译单个碎片行。")
        if context:
            prompt = (f"{instr}。以下是参考上下文(仅帮助理解整句语义，不要输出这些行):\n"
                      f"{context}\n\n需要翻译的内容(逐行对应输出，行数必须与输入一致):\n{text}")
        else:
            prompt = f"{instr}。原文:\n{text}"
        for attempt in range(max_retries):
            try:
                kwargs = dict(model=self.model,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}],
                              max_tokens=self.max_tokens)
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if self.temperature is not None and "temperature" in str(e).lower():
                    self.temperature = None   # 不支持 temperature 的模型 (如 reasoner): 去掉重试
                    continue
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5)
        raise RuntimeError("unreachable")

    def translate_lines(self, lines: list[str], target_lang: str = "Chinese",
                        batch_size: int = 10, max_retries: int = 2) -> list[str]:
        """分批翻译并强制行对齐 (字幕逐句场景)

        优先级链: 批量主引擎 → 逐条主引擎 → 本地兜底 (fallback) → 保留原文.
        兜底与保留原文均不中断整个任务 (供 UI 人工修正). 返回与输入等长的译文列表"""
        result: list[str] = []
        prev_ctx: list[str] = []   # 上一批末尾 2 行, 作为本批翻译的参考上下文 (解决批边界碎片)
        for start in range(0, len(lines), batch_size):
            chunk = lines[start:start + batch_size]
            joined = "\n".join(chunk)
            ctx_text = "\n".join(prev_ctx) if prev_ctx else None
            parts: list[str] = []
            batch_ok = False
            for _ in range(max_retries):
                try:
                    translated = self.translate_text(joined, target_lang, context=ctx_text)
                except Exception:
                    continue        # 批量失败: 重试
                parts = [ln.strip() for ln in translated.split("\n") if ln.strip()]
                if len(parts) == len(chunk):
                    batch_ok = True
                    break
                parts = []
            if not batch_ok:   # 批量失败或行数不齐: 逐条主引擎 → 本地兜底 → 保留原文
                parts = []
                fb_items: list[tuple[int, str]] = []
                for ln in chunk:
                    try:
                        parts.append(self.translate_text(ln, target_lang).strip())
                    except Exception:
                        parts.append(None)          # 占位, 交给兜底
                        fb_items.append((len(parts) - 1, ln))
                if fb_items and self.fallback is not None:
                    try:
                        fb_texts = self.fallback.translate_text(
                            [t for _, t in fb_items], target_lang)
                        for (idx, _), t in zip(fb_items, fb_texts):
                            parts[idx] = t.strip()
                    except Exception:
                        pass                        # 兜底失败 -> 保留原文
                parts = [p if p is not None else ln for p, ln in zip(parts, chunk)]
            result.extend(parts)
            prev_ctx = chunk[-2:]   # 记录本批末尾, 供下一批参考上下文
        if len(result) < len(lines):       # 兜底补齐
            result += [""] * (len(lines) - len(result))
        return result[:len(lines)]

    def translate_subtitle(self, srt_path: str, out_path: str,
                           target_lang: str = "Chinese") -> int:
        """翻译字幕文件(SRT/ASS), 保持时间轴不变. 返回失败行数"""
        import pysubs2
        subs = pysubs2.load(srt_path, encoding="utf-8")
        texts = [(i, e.plaintext) for i, e in enumerate(subs) if e.text.strip()]
        failed = 0
        batch_size = 20
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            lines = "\n".join(t for _, t in batch)
            try:
                translated = self.translate_text(lines, target_lang)
                parts = translated.split("\n")
                for (idx, _), part in zip(batch, parts):
                    subs[idx].text = part.strip()
            except Exception:
                failed += len(batch)
        subs.save(out_path, encoding="utf-8")
        return failed


# 向后兼容别名 (旧代码/旧配置仍可用)
DeepSeekTranslator = OpenAICompatTranslator


class LocalFallbackTranslator:
    """本地离线翻译兜底: Helsinki-NLP/opus-mt-en-zh (transformers pipeline)

    免 key、无需网络 (模型首次自动下载 ~300MB, 可用 HF_ENDPOINT=https://hf-mirror.com 镜像);
    质量中低, 仅作主引擎失败时的应急; 仅支持 en→zh (目标语言非中文时无法兜底, 会抛错)."""

    MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("VP_TRANSLATE_LOCAL_MODEL", self.MODEL_NAME)
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline
            print(f"[翻译] 加载本地兜底模型 {self.model_name} (首次自动下载 ~300MB)")
            self._pipe = pipeline("translation", model=self.model_name)

    def translate_text(self, text, target_lang: str = "Chinese",
                       source_lang: Optional[str] = None, max_retries: int = 1):
        """text: str 或 list[str]; 返回 str 或 list[str] (仅 en→zh)"""
        if isinstance(text, str):
            return self._translate_one(text, target_lang)
        return [self._translate_one(t, target_lang) for t in text]

    def _translate_one(self, text: str, target_lang: str) -> str:
        if target_lang not in ("Chinese", "中文"):
            raise RuntimeError(f"本地兜底模型仅支持 en→zh, 无法兜底目标语言 {target_lang}")
        self._ensure()
        return self._pipe(text)[0]["translation_text"].strip()


def get_translator(api_key: Optional[str] = None, base_url: Optional[str] = None,
                   model: Optional[str] = None,
                   fallback: Optional[LocalFallbackTranslator] = None) -> OpenAICompatTranslator:
    """运行时获取翻译器: 主引擎 (OpenAI 兼容, 支持实例级配置) + 本地兜底.

    无 key 时不报错 — translate_lines 会自动降级本地兜底 (迁移部署开箱即用)"""
    return OpenAICompatTranslator(api_key=api_key, base_url=base_url, model=model,
                                  fallback=fallback)
