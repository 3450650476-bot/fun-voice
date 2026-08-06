"""Fun-Voice 视频翻译配音工作台"""

import sys

# 中文日志在非 UTF-8 控制台 (cp1252/GBK 等) 直接 print 会 UnicodeEncodeError 崩溃,
# 且部分 except Exception 分支会把崩溃当业务异常吞掉 (如 _clip_ref_audio 静默返回原路径)。
# → 保留原编码, 但把不可编码字符替换为 ?, 保证任何环境下日志打印都不抛异常。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):  # 无 reconfigure 的流 (如重定向/某些 IDE) 或已关闭
        pass
