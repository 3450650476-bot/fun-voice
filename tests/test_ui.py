"""UI 修复轮回归测试: 日志彩色化渲染 / 进度条徽章 / build_app 可构造"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.ui as U

class TestRenderLogs(unittest.TestCase):
    """M1 阶段日志彩色化: (stage, msg) -> 带 class 的 HTML 行, 特殊字符转义"""

    def test_stage_class_per_line(self):
        # 进行中=绿(fv-log-cur), 已完成=蓝(fv-log-done) — 与胶囊颜色语义一致
        h = U.render_logs([(1, "提取完成"), (2, "分离完成"), (3, "ASR 完成")], current_stage=3)
        self.assertIn('class="fv-log-done"', h)      # 1/2 已完成
        self.assertIn('class="fv-log-cur"', h)       # 3 进行中
        self.assertIn("[1/7]", h)
        self.assertNotIn('fv-log-1', h)              # 不再按阶段固定色

    def test_done_all_blue(self):
        h = U.render_logs([(1, "a"), (2, "b")], current_stage=2, done=True)
        self.assertEqual(h.count('fv-log-done'), 2)
        self.assertEqual(h.count('fv-log-cur'), 0)

    def test_html_escaping(self):
        h = U.render_logs([(0, '失败: <x> & "y"')])
        self.assertIn('class="fv-log-0"', h)         # 异常=红
        self.assertIn("&lt;x&gt; &amp; &quot;y&quot;", h)
        self.assertNotIn("<x>", h)

    def test_empty(self):
        self.assertEqual(U.render_logs([]), "")


class TestFlowHtml(unittest.TestCase):
    """M2 进度条+阶段徽章: 未开始全 pend, 当前步 cur, 已完成 done, 进度条宽度按 stage/7"""

    def test_stage0_all_pending(self):
        f = U.flow_html(0)
        self.assertEqual(f.count("fv-pend"), 6)
        self.assertEqual(f.count("fv-cur"), 0)
        self.assertNotIn("fv-progress", f)      # 已移除进度条, 仅保留胶囊

    def test_stage3_marks_done_and_cur(self):
        f = U.flow_html(3)
        self.assertEqual(f.count("fv-done"), 2)      # 上传/人声分离
        self.assertEqual(f.count("fv-cur"), 1)       # 语音识别
        self.assertEqual(f.count("fv-pend"), 3)      # 翻译/克隆/混流
        self.assertNotIn("fv-progress", f)

    def test_stage7_all_done(self):
        f = U.flow_html(7)
        self.assertEqual(f.count("fv-done"), 6)
        self.assertEqual(f.count("fv-cur"), 0)
        self.assertNotIn("fv-progress", f)

    def test_flow_steps_contained(self):
        f = U.flow_html(1)
        for s in U.FLOW_STEPS:
            self.assertIn(s, f)


class TestSysStats(unittest.TestCase):
    """系统监控条: CUDA 状态 / 显存 / 内存 (mock torch.cuda)"""

    def test_cuda_path(self):
        with unittest.mock.patch.object(U.torch.cuda, "is_available", return_value=True), \
             unittest.mock.patch.object(U.torch.cuda, "get_device_name", return_value="RTX 3060"), \
             unittest.mock.patch.object(U, "_gpu_mem_mb", return_value=(2048.0, 12288.0)):
            h = U.sys_stats_html()
        self.assertIn("CUDA ✓ · RTX 3060", h)
        self.assertIn("2048 / 12288 MB", h)          # nvidia-smi 全局显存
        self.assertIn("width:17%", h)                # 2048/12288≈16.7% 四舍五入
        self.assertIn("内存", h)
        self.assertIn("GB", h)                        # 系统内存 (与任务管理器一致)

    def test_cpu_path(self):
        with unittest.mock.patch.object(U.torch.cuda, "is_available", return_value=False):
            h = U.sys_stats_html()
        self.assertIn("CPU（无 CUDA）", h)
        self.assertIn("显存", h)

    @unittest.skipUnless(sys.platform == "win32", "仅 Windows: 读取本机内存")
    def test_mem_reads_windows(self):
        m = U._process_mem_mb()
        self.assertGreater(m, 0)                  # Windows 下真实读取成功


class TestBuildApp(unittest.TestCase):
    """UI 修复轮结构回归: build_app 可构造 (参数重排/19 元组/HTML 日志不破坏构建)"""

    def test_build_app_constructs(self):
        demo = U.build_app()
        self.assertIsNotNone(demo)
        # 关键组件存在 (通过 elem_id 定位)
        ids = {c.elem_id for c in demo.blocks.values() if getattr(c, "elem_id", None)}
        for need in ("fv-log", "fv-flow-progress", "fv-params-group", "fv-sys"):
            self.assertIn(need, ids)

    def test_output_arity_matches(self):
        """build_output 20 元组 与 事件 outputs 列表长度一致 (AST 源码检查)"""
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "app", "ui.py")
        tree = ast.parse(open(path, encoding="utf-8").read())

        def build_output_returns():
            n = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "build_output":
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
                            n.append(len(sub.value.elts))
            return n

        self.assertEqual(build_output_returns(), [20, 20, 20])
        arities = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "click" and len(node.args) >= 3
                    and isinstance(node.args[2], ast.List)):
                arities[ast.unparse(node.func)] = len(node.args[2].elts)
        self.assertEqual(arities["run_btn.click"], 20)
        self.assertEqual(arities["resume_btn.click"], 20)
        self.assertEqual(arities["stop_btn.click"], 2)
        self.assertEqual(arities["save_btn.click"], 10)


if __name__ == "__main__":
    unittest.main()
