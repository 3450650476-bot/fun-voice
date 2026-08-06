"""Fun-Voice 回归测试套件

运行: cd fun-voice && PYTHONUTF8=1 ./.venv/Scripts/python.exe -m unittest discover -s tests -v
全部 fake/mock, 不加载真实模型权重, 不依赖 ffmpeg/sox/网络, 秒级完成.
覆盖已修复回归: B1 pad 边界 / B2 显存释放 / A1-A5 音轨 / B5 rerun 参数 /
B6 翻译容错 / T1-T3 翻译 / T2/T3/T6/T8 与 batch_size / 自动模式移除 / T7 yield 序列.
"""
