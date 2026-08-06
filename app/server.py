"""Fun-Voice 启动脚本: python -m app.server

启动前自动清理 gradio 上传缓存 (旧会话残留文件会导致重启后浏览器报
PermissionError / 引用失效); 然后启动 Web UI (默认 8080 端口, 自动开浏览器)
"""
import os
import shutil
import tempfile

import gradio as gr

from app.ui import build_app, CELEB_ROOT, UI_CSS

# gradio 白名单: UI 组件(名人头像 Gallery/试听 Audio)直接引用模型目录文件,
# 必须加入 allowed_paths 否则 gradio 6 报 InvalidPathError
ALLOWED_PATHS = [CELEB_ROOT]

# ⚠️ 强制深色主题: 用自定义深色主题替换默认 Soft(浅色) 主题
# 根因: gradio 前端 JS 会把 var(--body-background-fill) 内联写入容器背景,
# 而 Soft 主题的背景变量是浅色 neutral_50 → 页面底部/容器底色发白。
# 从主题源头就设置深色值, 前端写入的就是深色 (这是根治, 之前的 CSS/JS 覆盖都是打地鼠)
DARK_THEME = gr.themes.Base()
DARK_THEME.set(
    background_fill_primary="#101010",
    background_fill_secondary="#171717",
    block_background_fill="#1f1f1f",
    body_background_fill="#101010",
    block_border_color="#3d3d3d",
    block_border_color_dark="#3d3d3d",
    body_text_color="#d4d4d4",
    body_text_color_dark="#d4d4d4",
    block_label_text_color="#c8c8c8",
    block_title_text_color="#c8c8c8",
    # ⚠️ 字幕表格: gradio 6 数据行背景用 var(--table-even/odd-background-fill) 且自带 !important,
    #    元素级 CSS 无法覆盖 → 必须从主题变量源头钉死 (表头/内容统一黑底白字)
    table_even_background_fill="#000000",
    table_even_background_fill_dark="#000000",
    table_odd_background_fill="#000000",
    table_odd_background_fill_dark="#000000",
    table_text_color="#ffffff",
    table_text_color_dark="#ffffff",
    table_border_color="#3d3d3d",
    table_border_color_dark="#3d3d3d",
    table_row_focus="#2d2d2d",
    table_row_focus_dark="#2d2d2d",
)

# ⚠️ 强制深色 (无视系统/浏览器深浅色偏好): 页面加载即锁定深色
# 之前用 gr.HTML script 不可靠 (gradio 6 会剥离/不触发),
# 改走 launch(js=) 全局注入 — 这是 gradio 保证执行的前端 JS 通道
# (界面主题色切换已删除 2026-08-05: 页面固定深色方案)
FORCE_DARK_JS = """
() => {
  // gradio 前端运行时会给容器内联设置 background:var(--body-background-fill) 并切换 body dark class
  // 容器是 JS 渲染后才创建的 → 必须监听 childList 且每次重新查找, 用字面量钉死
  const applyDark = () => {
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-theme', 'dark');
    document.body.classList.add('dark');
    document.body.classList.add('fv-t-dark');
    document.body.style.background = '#101010';
    document.body.style.color = '#d4d4d4';
    const root = document.querySelector('.gradio-container');
    if (root) {
      root.style.background = '#171717';
      root.style.color = '#d4d4d4';
      root.style.setProperty('--body-background-fill', '#171717');
      root.style.setProperty('--body-background-fill-dark', '#171717');
      root.style.setProperty('--background-fill-primary', '#171717');
      root.style.setProperty('--background-fill-secondary', '#1f1f1f');
      root.style.setProperty('--color-accent', '#378ADD');
    }
    const footer = document.querySelector('[data-testid="footer"], .footer, footer');
    if (footer) footer.style.background = '#101010';
  };
  applyDark();
  new MutationObserver(applyDark).observe(document.documentElement,
      {attributes: true, childList: true, subtree: true});
  return [];
}
"""


def _clean_gradio_cache():
    """清理 gradio 上传缓存目录 Temp\\gradio (仅删子项, 保留 vibe_edit_history)

    原因: gradio 把上传文件复制到 Temp\\gradio\\<hash>\\, 服务器重启后旧文件
    被浏览器页面引用时可能报 PermissionError; 每次启动清一遍最干净.
    """
    cache = os.path.join(tempfile.gettempdir(), "gradio")
    if not os.path.isdir(cache):
        return 0
    removed = 0
    for name in os.listdir(cache):
        p = os.path.join(cache, name)
        if name == "vibe_edit_history":
            continue
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            removed += 1
        except Exception:
            pass
    return removed


if __name__ == "__main__":
    n = _clean_gradio_cache()
    if n:
        print(f"[server] cleaned gradio cache: {n} item(s)")
    app = build_app()
    # 强制深色 - 源头级修复:
    # ① theme_mode="dark" → 底部设置面板默认勾选深色
    # ② body_css 四件套全深色 → gradio 前端 JS 内联写入容器的背景就是深色,
    #    根治"页面最底部/容器底色是浅色 #f9fafb" (默认浅色, 之前靠 JS 事后覆盖是打地鼠)
    _orig_get_config = app.get_config_file

    def _patched_get_config():
        cfg = _orig_get_config()
        cfg["theme_mode"] = "dark"
        if cfg.get("body_css"):
            cfg["body_css"] = {
                "body_background_fill": "#101010",
                "body_text_color": "#d4d4d4",
                "body_background_fill_dark": "#101010",
                "body_text_color_dark": "#d4d4d4",
            }
        return cfg

    app.get_config_file = _patched_get_config
    # 显存串行: 并发必须为 1 (MDX/ASR/TTS 三个模型轮流占用 GPU)
    # theme=DARK_THEME: 从主题源头就是深色, 根治浅色背景
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=8080, inbrowser=True,
        theme=DARK_THEME, allowed_paths=ALLOWED_PATHS, css=UI_CSS,
        js=FORCE_DARK_JS)
