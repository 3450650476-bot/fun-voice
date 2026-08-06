"""Fun-Voice 视频配音工作台 - Gradio Web UI

Tab1 输入参数: 视频上传 / 参考音色三来源 / 参数 / 运行+7步进度
Tab2 字幕编辑: 只读表格 + 下方面板逐条编辑 + 增量重跑
Tab3 结果下载: 视频预览 + 4 下载按钮
"""
from __future__ import annotations

import os
from pathlib import Path

import torch   # 先于 gradio import, 保证 CUDA DLL 加载顺序 (与 server.py 同款)

# ⚠️ 必须先加载 torch (通过 app.pipeline) 再 import gradio: DLL 加载顺序冲突
# gradio 先加载会导致 Segmentation fault (torch 的 cuDNN/OpenCV DLL 冲突)
from app.pipeline import Pipeline, PipelineResult

import gradio as gr

# 模型根目录: 优先 VP_MODELS 环境变量, 缺省相对项目根定位 (K:\视频翻译与配音\models)
MODELS_ROOT = os.environ.get("VP_MODELS", str(Path(__file__).resolve().parents[2] / "models"))
CELEB_ROOT = os.path.join(MODELS_ROOT, "celebrities30s")
STEP_NAMES = ["提取音轨", "人声分离", "语音识别", "翻译", "克隆配音", "时间轴对齐", "混流出片"]


def scan_celebrities() -> dict[str, dict]:
    """扫描名人音色库 -> {显示名: {"mp3": 音频路径, "jpg": 头像路径或None}}"""
    items: dict[str, dict] = {}
    if not os.path.isdir(CELEB_ROOT):
        return items
    for lang in sorted(os.listdir(CELEB_ROOT)):
        lp = os.path.join(CELEB_ROOT, lang)
        if not os.path.isdir(lp):
            continue
        for f in sorted(os.listdir(lp)):
            if f.lower().endswith(".mp3"):
                name = f[:-4]
                jpg = os.path.join(lp, name + ".jpg")
                items[f"{lang} · {name}"] = {
                    "mp3": os.path.join(lp, f),
                    "jpg": jpg if os.path.exists(jpg) else None,
                }
    return items


def fmt_ts(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# ==================== 全局样式: 强制深色 + 主题色切换 + 美化 ====================
# ⚠️ 强制深色: 关键变量全部 !important, 不受 gradio theme_mode / 系统深浅色影响
UI_CSS = r"""
/* ---- 强制深色 (变量级钉死 + body 兜底 + 容器字面量背景) ---- */
html, body{background:#101010 !important;color:#d4d4d4 !important}
/* 字面量背景: 压过 gradio JS 内联写入的 var(--body-background-fill) 解析结果 */
.gradio-container{background:#171717 !important}
.gradio-container {
  --background-fill-primary:#171717 !important;
  --body-background-fill:#171717 !important;
  --block-background-fill:#1f1f1f !important;
  --stat-background-fill:#262626 !important;
  --body-text-color:#d4d4d4 !important;
  --body-text-color-subdued:#9a9a9a !important;
  --block-label-text-color:#c8c8c8 !important;
  --border-color-primary:#3d3d3d !important;
  --border-color-secondary:#333333 !important;
  --ui-background-color:#262626 !important;
  --ui-background-color-hover:#2e2e2e !important;
  --ui-foreground-color:#d4d4d4 !important;
  --color-accent:#378ADD !important;
  --color-accent-soft:#1e3a5f !important;
  font-family:'Microsoft YaHei','PingFang SC','Segoe UI',sans-serif;
}
/* 输入/文本域统一深色 */
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background:#262626 !important;
  color:#d4d4d4 !important;
  border-color:#3d3d3d !important;
}
.gradio-container textarea { overflow-y:auto !important; }
.gradio-container textarea::-webkit-scrollbar{width:8px}
.gradio-container textarea::-webkit-scrollbar-thumb{background:#555;border-radius:4px}
/* 表格/视频/音频深色 */
.gradio-container table { background:#1f1f1f !important; }
/* ---- 主题色切换 (整体背景+强调色, JS 切换 body class 即时生效; !important 同权重压过基础深色) ---- */
body.fv-t-dark   .gradio-container{--color-accent:#4a90d9 !important;--color-accent-soft:#1c3550 !important;
  --background-fill-primary:#101010 !important;--body-background-fill:#101010 !important;
  --block-background-fill:#181818 !important;--ui-background-color:#1e1e1e !important}
body.fv-t-blue   .gradio-container{--color-accent:#378ADD !important;--color-accent-soft:#1e3a5f !important;
  --background-fill-primary:#101826 !important;--body-background-fill:#101826 !important;
  --block-background-fill:#16213a !important;--ui-background-color:#1a2745 !important}
body.fv-t-teal   .gradio-container{--color-accent:#1D9E75 !important;--color-accent-soft:#0f3d2e !important;
  --background-fill-primary:#0f1d18 !important;--body-background-fill:#0f1d18 !important;
  --block-background-fill:#162a22 !important;--ui-background-color:#1a3027 !important}
body.fv-t-purple .gradio-container{--color-accent:#7F77DD !important;--color-accent-soft:#2a2458 !important;
  --background-fill-primary:#171430 !important;--body-background-fill:#171430 !important;
  --block-background-fill:#211c40 !important;--ui-background-color:#262050 !important}
body.fv-t-orange .gradio-container{--color-accent:#D85A30 !important;--color-accent-soft:#4a1f0d !important;
  --background-fill-primary:#1f130d !important;--body-background-fill:#1f130d !important;
  --block-background-fill:#2b1a12 !important;--ui-background-color:#33201a !important}
/* ---- Tab 标签: 圆角胶囊形 (未选中=描边弱化, 选中=强调色填充) ---- */
.gradio-container .tab-nav,
.gradio-container [data-testid="tab-nav"]{
  gap:8px !important;
}
.gradio-container .tab-nav button,
.gradio-container [data-testid="tab-nav"] button{
  font-size:18px !important; font-weight:600 !important;
  padding:8px 24px !important;
  border-radius:22px !important;
  border:1px solid var(--border-color-primary) !important;
  background:transparent !important;
  color:var(--body-text-color-subdued) !important;
  transition:border-color .15s ease,color .15s ease,background .15s ease;
}
.gradio-container .tab-nav button:hover,
.gradio-container [data-testid="tab-nav"] button:hover{
  border-color:var(--color-accent) !important;
  color:var(--color-accent) !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container [data-testid="tab-nav"] button.selected{
  border-color:var(--color-accent) !important;
  background:var(--color-accent) !important;
  color:#fff !important;
  border-bottom-color:var(--color-accent) !important;
}
/* ---- 运行日志填满底部 + 标题居中 ---- */
#fv-log textarea{min-height:260px}
#fv-log [data-testid="block-info"]{text-align:center !important;width:100% !important}
/* ---- Tab2 全文浏览: 原文/译文标题居中 ---- */
#fv-full-orig [data-testid="block-info"],
#fv-full-zh [data-testid="block-info"]{text-align:center !important;width:100% !important}
/* ---- Tab2 字幕表格: 表头与内容底纹严格同色, 黑底白字 ----
   核心: gradio 6 数据行背景 var(--table-even/odd-background-fill) 自带 !important,
   元素级规则赢不了 → 这里在容器上重定义主题变量(带 !important) 从源头覆盖;
   下面 table/th/td 规则保留作兜底 */
.gradio-container{
  --table-even-background-fill:#000000 !important;
  --table-even-background-fill_dark:#000000 !important;
  --table-odd-background-fill:#000000 !important;
  --table-odd-background-fill_dark:#000000 !important;
  --table-text-color:#ffffff !important;
  --table-text-color_dark:#ffffff !important;
  --table-border-color:#3d3d3d !important;
  --table-border-color_dark:#3d3d3d !important;
  --table-row-focus:#2d2d2d !important;
  --table-row-focus_dark:#2d2d2d !important;
}
.gradio-container table,
.gradio-container table th,
.gradio-container table th.header-cell,
.gradio-container table thead,
.gradio-container table td{
  background:#000000 !important;
  color:#ffffff !important;
  border-color:#3d3d3d !important;
}
/* ---- 参数设置主框 (单主边框, 包住五个参数) / 下载卡片 ---- */
#fv-params-group{
  border:1px solid var(--border-color-primary) !important;
  background:var(--block-background-fill) !important;
  border-radius:12px !important;
  margin-bottom:12px !important;
  padding:10px 14px !important;
}
#fv-dl-group{
  border-radius:12px !important;
  margin-bottom:12px !important;
}
#fv-params-group .column,
#fv-dl-group .column{padding:0 !important}
/* 下载按钮: 2x2 等宽网格 + hover 高亮 */
#fv-dl-group .downloadbutton button,
#fv-dl-group button{width:100% !important;margin:5px 0 !important}
/* ---- 组件标题(block-info): 统一 15px/600, 独占一行 ---- */
.gradio-container [data-testid="block-info"]{
  font-size:15px !important;font-weight:600 !important;
  color:var(--body-text-color) !important;
  display:block !important;
}
/* ---- 卡片化 + 间距美化 ---- */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  border-radius:12px;
}
.gradio-container { font-size:14px; }
.gradio-container h1{font-size:20px;font-weight:600}
.gradio-container h2{font-size:17px;font-weight:600}
.gradio-container h3{font-size:15px;font-weight:600}
.gradio-container button{
  border-radius:8px !important;
  font-weight:500;
}
.gradio-container .primary,
.gradio-container button.primary{
  background:var(--color-accent) !important;
  border-color:var(--color-accent) !important;
  color:#fff !important;
}
/* ---- 响应式: 窄屏两栏变一栏 ---- */
@media (max-width:900px){
  .gradio-container .wrap,
  .gradio-container [data-testid="row"]{ flex-direction:column !important; }
}

/* ==================== 美化2.0 (品牌区/空状态/按钮层级/动效/焦点环/hover/间距/字体) ==================== */
/* 1. 顶部品牌区: 大标题 + 流程说明 + 细分隔线, 与状态/主题区视觉分离 */
#fv-topbar{
  border-bottom:1px solid var(--border-color-primary);
  padding:2px 0 12px !important;
  margin-bottom:8px !important;
}
#fv-brand h2{
  font-size:24px !important;font-weight:700 !important;
  letter-spacing:.5px;margin:0 0 4px !important;
}
#fv-flow .prose p{font-size:13px;color:var(--body-text-color-subdued);margin:2px 0 0}
#fv-flow .prose strong{color:var(--color-accent);font-weight:600}
#fv-env .prose p{font-size:12px;color:var(--body-text-color-subdued);text-align:right;margin:2px 0 4px}
#fv-env .prose strong{color:var(--body-text-color)}
/* 2. 空状态引导 (Tab2 未运行时教学提示) */
#fv-empty{
  text-align:center;color:var(--body-text-color-subdued);
  border:1px dashed var(--border-color-primary);
  border-radius:12px;padding:26px 16px;margin:6px 0 14px;
}
#fv-empty .prose p{color:var(--body-text-color-subdued);margin:4px 0}
#fv-empty .prose strong{color:var(--color-accent)}
/* 3. 按钮层级: 停止按钮幽灵样式 (描边, 弱于主按钮) */
#fv-stop-btn{
  background:transparent !important;
  border:1px solid var(--border-color-primary) !important;
  color:var(--body-text-color-subdued) !important;
}
#fv-stop-btn:hover{
  border-color:var(--color-accent) !important;
  color:var(--color-accent) !important;
  background:transparent !important;
}
/* 4. 完成动效: 结果卡片淡入上浮 150ms */
@keyframes fv-pop{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.fv-pop{animation:fv-pop .15s ease-out}
/* 5. 键盘焦点环 (无障碍) */
.gradio-container :focus-visible{
  outline:2px solid var(--color-accent) !important;
  outline-offset:2px;border-radius:4px;
}
/* 6. 音色库头像 hover: 高亮 + 轻微上浮 */
.gradio-container [data-testid="gallery"] img,
.gradio-container .gallery img{
  transition:transform .15s ease,filter .15s ease;
  cursor:pointer;
}
.gradio-container [data-testid="gallery"] figure:hover img,
.gradio-container .gallery figure:hover img{
  transform:translateY(-3px);filter:brightness(1.12);
}
/* 7. 间距节奏: 组件分级 (组内紧凑/区块宽松) */
.gradio-container .form{margin:4px 0 10px !important}
.gradio-container .block{padding:6px 0}
/* 8. 字体层级: 页内小标题统一 14px 加粗 */
.gradio-container h5{font-size:14px !important;font-weight:600 !important;color:var(--body-text-color) !important}
.gradio-container .prose p{line-height:1.6}
.gradio-container .prose{font-size:14px}

/* ==================== 美化2.1 (单框参数区/留白/主题横排/流程徽章/下载按钮) ==================== */
/* ---- 参数设置: 内层控件无独立边框, 全部纳入 panel 主框 ---- */
/* 1. 控件容器彻底无边框/背景 (Radio 选项/Dropdown/Slider 全部) */
#fv-params-group [role="radiogroup"],
#fv-params-group .wrap,
#fv-params-group .form,
#fv-params-group .block{
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
}
/* 2. 参数之间留白节奏: 只留上下间距, 无左右 margin (避免 scrollWidth 溢出产生横移滑条) */
#fv-params-group .block{margin:14px 0 18px !important;padding:0 4px !important}
#fv-params-group .form{margin:0 !important;padding:0 !important}
#fv-params-group .wrap{margin:0 !important;padding:0 4px !important}
/* Slider 结构容器 (.wrap > .head): 不参与留白, 保证标题与 Radio 标题左对齐 */
#fv-params-group .wrap:has(> .head){margin:0 !important;padding:0 !important}
/* 3. 单选按钮: 完全纳入主框内 + 选中圆点亮红 */
.gradio-container{
  --checkbox-border-color-selected:#ff3b30 !important;
  --checkbox-background-color-selected:#ff3b30 !important;
  --checkbox-background-color-selected-hover:#ff3b30 !important;
  --checkbox-border-color-focus:#ff3b30 !important;
}
.gradio-container input[type="radio"]{
  width:16px;height:16px;cursor:pointer;flex-shrink:0;
  vertical-align:middle;
}
.gradio-container input[type="radio"]:checked{
  border-color:#ff3b30 !important;
}
.gradio-container input[type="radio"]:focus{
  border-color:#ff3b30 !important;
  box-shadow:0 0 0 2px rgba(255,59,48,.25) !important;
}
/* 4. 未选中单选文字黑色; 选中项文字亮红 */
.gradio-container label:has(input[type="radio"]){color:#000 !important}
.gradio-container label:has(input[type="radio"]:checked){
  color:#ff3b30 !important;font-weight:600;
}
/* 5. 滑块 + 填写框: 滑块行加高 (min-height 44px), min/滑块/max 垂直居中, 填写框 absolute 对齐滑块行 */
#fv-params-group .wrap:has(> .head){position:relative !important;display:block !important}
#fv-params-group .wrap:has(> .head) .head{display:block !important}
#fv-params-group .wrap:has(> .head) .slider_input_container{
  display:flex !important;align-items:center !important;position:relative !important;
  min-height:44px !important;   /* 增大滑块区域高度 */
}
#fv-params-group .wrap:has(> .head) .min_value{
  width:26px !important;text-align:center !important;flex:0 0 auto !important;
}
#fv-params-group .wrap:has(> .head) input[type="range"]{width:62% !important;flex:0 0 auto !important}
#fv-params-group .wrap:has(> .head) .max_value{
  width:30px !important;text-align:center !important;flex:0 0 auto !important;
}
#fv-params-group .wrap:has(> .head) .tab-like-container{
  position:absolute !important;
  left:calc(62% + 76px) !important;   /* min(26) + 间距(10) + max(30) + 间距(10) = 76 */
  top:calc(100% - 34px) !important;   /* 对齐滑块行(高44px)垂直中心: 44/2 + 数字框半高16.5 - 微调 */
  display:flex !important;align-items:center;gap:6px !important;
}
#fv-params-group .tab-like-container .reset-button{display:none !important}
#fv-params-group .tab-like-container input[type="number"]{
  width:56px !important;height:33px !important;   /* 高度与人声分离胶囊(33px)一致 */
  background:#262626 !important;color:#d4d4d4 !important;
  border:1px solid var(--border-color-primary) !important;
  border-radius:6px !important;
  font-size:13px !important;text-align:center !important;
  -moz-appearance:textfield !important;
  -webkit-appearance:none !important;   /* 彻底隐藏上下箭头(spinner) */
  appearance:none !important;
}
/* 隐藏 number 输入框的上下箭头(spinner) — 会显示成竖向箭头条, 像竖向滑动框 */
#fv-params-group .tab-like-container input[type="number"]::-webkit-inner-spin-button,
#fv-params-group .tab-like-container input[type="number"]::-webkit-outer-spin-button{
  -webkit-appearance:none !important;
  display:none !important;
  margin:0 !important;
}
/* 开始/停止按钮缩小 */
#fv-params-group .primary,
#fv-params-group button.primary,
#fv-stop-btn{padding:6px 18px !important;font-size:13.5px !important}
/* 参考文本: 标题文字居中 */
#fv-ref-text [data-testid="block-info"]{text-align:center !important;width:100% !important}
/* 上传参考音频: 与上方(视频/音色模式)拉开空间 */
#fv-ref-audio{margin-top:16px !important}
/* 参数区五个标题(人声分离/目标语言/配音拉伸上限/配音音量/成品画质): 统一醒目格式 + 左对齐 */
#fv-params-group [data-testid="block-info"]{
  font-size:15.5px !important;font-weight:700 !important;
  letter-spacing:.3px;margin:0 0 6px !important;
  text-align:left !important;
}
/* 音色库人名标签: 无边框 + 半透明白底黑字 (gradio 6 人名标签是 .caption-label) */
.gradio-container .caption-label{
  color:#000 !important;
  background:rgba(255,255,255,.65) !important;
  border:none !important;
  font-size:12.5px !important;
  font-weight:600 !important;
  text-align:center !important;
  border-radius:4px !important;
}
/* 配音音色(上传音频/音色库)双胶囊: 等宽 + 与"开始配音"按钮同尺寸 (8px 圆角/6x18px/13.5px/无边框=高度32px), 选中填充强调色 */
#fv-ref-mode{margin-top:20px !important}
#fv-ref-mode .wrap{display:flex !important;gap:10px !important;margin:0 !important;padding:0 !important}
#fv-ref-mode .wrap label,
#fv-ref-mode label[data-testid$="-radio-label"]{
  flex:1 !important;
  border-radius:8px !important;
  padding:6px 18px !important;
  font-size:13.5px !important;
  justify-content:center !important;
  background:var(--ui-background-color) !important;
  border:none !important;
  color:var(--body-text-color) !important;
  font-weight:500 !important;
}
#fv-ref-mode input[type="radio"]{display:none !important}
#fv-ref-mode label:has(input[type="radio"]:checked){
  background:var(--color-accent) !important;
  color:#fff !important;
}
/* 顶部流程: 徽章式步骤条 (6 胶囊统一 accent 高亮配色) */
.fv-flow{display:flex;align-items:center;flex-wrap:wrap;gap:5px;margin:7px 0 0}
.fv-step{
  background:var(--color-accent-soft);
  border:1px solid var(--color-accent);
  border-radius:14px;padding:2px 11px;
  font-size:12.5px;color:var(--color-accent);
  white-space:nowrap;
}
.fv-arrow{color:var(--body-text-color-subdued);font-size:12px}
/* 下载按钮: 卡片式按钮 (图标+文字, hover 高亮) */
#fv-dl-group button{
  padding:13px 10px !important;border-radius:10px !important;
  background:var(--block-background-fill) !important;
  border:1px solid var(--border-color-primary) !important;
  color:var(--body-text-color) !important;
  font-size:14px;font-weight:500;
  transition:border-color .15s ease,color .15s ease,transform .15s ease;
}
#fv-dl-group button:hover{
  border-color:var(--color-accent) !important;
  color:var(--color-accent) !important;
  background:var(--ui-background-color-hover) !important;
  transform:translateY(-1px);
}
/* ==================== 阶段日志彩色化 + 进度条/徽章高亮 (UI 修复轮) ==================== */
/* 日志 HTML 容器: 滚动 + 行距 */
#fv-log{min-height:240px;overflow-y:auto;border:1px solid var(--border-color-primary);
  border-radius:10px;background:#1b1b1b !important;padding:8px 12px}
#fv-log [data-testid="block-info"]{text-align:center !important;width:100% !important}
#fv-log [data-testid="html"] div{margin:3px 0;font-size:13px;line-height:1.5;
  font-family:Consolas,'Microsoft YaHei',monospace;white-space:pre-wrap;word-break:break-all}
/* 每阶段配色 (提取=蓝/分离=绿/ASR=橙/翻译=紫/TTS=粉/对齐=青/混流=黄/异常=红) */
/* 日志颜色与胶囊一致: 进行中=绿(fv-log-cur), 已完成=蓝(fv-log-done), 异常/停止=红(fv-log-0) */
.fv-log-cur{color:#7ad0a0 !important}
.fv-log-done{color:#7fb2e5 !important}
.fv-log-0{color:#ff6b6b !important}
/* 流程徽章状态: 进行中=绿, 已完成=蓝, 未开始=弱化 (与日志一致) */
.fv-step.fv-cur{background:#1d4a2e !important;border-color:#2e7d4f !important;color:#7ad0a0 !important;font-weight:700}
.fv-step.fv-done{background:#14304a !important;border-color:#2e5e8a !important;color:#7fb2e5 !important}
.fv-step.fv-pend{opacity:.4}
/* ---- 系统监控条 (日志上方实时刷新: CUDA/显存/内存) ---- */
#fv-sys{border:1px solid var(--border-color-primary);border-radius:10px;
  background:#1b1b1b !important;padding:6px 14px;margin:4px 0 8px;font-size:13px}
#fv-sys [data-testid="html"] div{display:flex;flex-wrap:wrap;gap:16px;align-items:center}
.fv-sys-item{color:var(--body-text-color-subdued);white-space:nowrap}
.fv-sys-item b{color:#d4d4d4;font-weight:600}
.fv-minibar{display:inline-block;width:64px;height:6px;background:#333 !important;
  border-radius:3px;overflow:hidden;vertical-align:middle;margin-left:6px}
.fv-minibar i{display:block;height:100%;background:var(--color-accent) !important;
  border-radius:3px;transition:width .4s ease}
.fv-minibar.warn i{background:#ff6b6b !important}
/* ---- 隐藏 gradio 页脚(通过 API 使用/使用 Gradio 构建)与右上角设置按钮 ---- */
footer,
[data-testid="footer"],
.footer{display:none !important}
button[aria-label="Settings"],
button[aria-label="设置"],
[data-testid="settings"]{display:none !important}
"""

# (界面主题色切换已删除 2026-08-05: 页面固定深色方案)

def info_card_html(drift: float, workspace: str, done: bool = False,
                   timings: dict | None = None) -> str:
    """结果信息卡片 (深色卡片样式, 完成时淡入上浮动效)
    done 且存在成品 → "✅ 配音完成"; done 但无成品 (如"仅到翻译"提前停止) → "⏹ 已停止";
    不显示 GPU 显存 (顶部监控条已实时展示, 2026-08-06)"""
    has_out = bool(done and workspace and os.path.exists(
        os.path.join(workspace, "07_output.mp4")))
    if has_out:
        color, mark = "#7fb2e5", "✅ 配音完成"
    elif done:
        color, mark = "#f0b46a", "⏹ 已停止（未混流出片）"
    else:
        color, mark = "#9a9a9a", "⏳ 处理中"
    anim = " fv-pop" if done else ""
    t_line = ""
    if timings:
        parts = " · ".join(f"{k}={v:.0f}s" for k, v in timings.items() if v > 0)
        if parts:
            t_line = (f'<div style="font-size:12px;color:#9a9a9a">阶段耗时</div>'
                      f'<div style="font-size:12px;color:#9a9a9a;word-break:break-all">{parts}</div>')
    return (f'<div class="{anim}" style="background:#262626;border:1px solid #3d3d3d;border-radius:12px;'
            f'padding:14px 16px">'
            f'<div style="font-size:12px;color:#9a9a9a">{mark} · 时间轴漂移</div>'
            f'<div style="font-size:24px;font-weight:600;color:#d4d4d4;margin:2px 0">{drift:.1f}s</div>'
            f'<div style="font-size:12px;color:#9a9a9a">产物目录</div>'
            f'<div style="font-size:12px;color:{color};word-break:break-all">{workspace}</div>'
            + t_line + '</div>')


# ==================== 日志/进度条渲染 (模块级纯函数, 可单测) ====================
import html as _html

FLOW_STEPS = ["上传视频", "人声分离", "语音识别", "原文翻译", "克隆配音", "混流出片"]


def _system_mem_mb() -> tuple[float, float]:
    """系统物理内存 (总, 可用) MB via GlobalMemoryStatusEx — 与任务管理器"内存"页一致"""
    try:
        import ctypes
        from ctypes import wintypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD),
                        ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return (ms.ullTotalPhys / 1024 ** 2, ms.ullAvailPhys / 1024 ** 2) if ok else (0.0, 0.0)
    except Exception:
        return 0.0, 0.0


def _process_mem_mb() -> float:
    """当前进程内存占用 MB (Windows via ctypes; 其他平台返回 -1)"""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        # 必须显式声明 argtypes/restype: 64 位下 HANDLE/SIZE_T 是 8 字节,
        # 默认按 c_int 传参会截断指针, 导致 GetProcessMemoryInfo 静默失败返回 0
        _psapi = ctypes.WinDLL("psapi")
        _k32 = ctypes.WinDLL("kernel32")
        _k32.GetCurrentProcess.restype = wintypes.HANDLE
        _psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        ok = _psapi.GetProcessMemoryInfo(
            _k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 1024 ** 2 if ok else -1.0
    except Exception:
        return -1.0


def _gpu_mem_mb() -> tuple[float, float] | None:
    """GPU 显存 (used, total) MB。优先 nvidia-smi 全局查询 (含其他进程),
    失败回退 torch.memory_allocated (仅本进程 PyTorch 分配, 会偏低)"""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        used, total = out.stdout.strip().split(",")[:2]
        return float(used), float(total)
    except Exception:
        try:
            return (torch.cuda.memory_allocated() / 1024 ** 2,
                    torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)
        except Exception:
            return None


def sys_stats_html() -> str:
    """系统监控卡片: CUDA 状态 / 显存占用(全局, 含占用条) / 进程内存 — 日志上方实时刷新"""
    cuda = torch.cuda.is_available()
    gpu, vram, vram_bar = "CPU（无 CUDA）", "-", ""
    if cuda:
        try:
            name = torch.cuda.get_device_name(0)
            mem = _gpu_mem_mb()
            if mem is None:
                raise RuntimeError("无法查询显存")
            alloc, total = mem
            gpu = f"CUDA ✓ · {name}"
            vram = f"{alloc:.0f} / {total:.0f} MB"
            if total > 0:
                pct = alloc / total * 100
                warn = " warn" if pct > 85 else ""
                vram_bar = f'<span class="fv-minibar{warn}"><i style="width:{pct:.0f}%"></i></span>'
        except Exception:
            gpu, vram = "CUDA ✓", "-"
    total_m, avail_m = _system_mem_mb()
    if total_m > 0:
        used_g, total_g = (total_m - avail_m) / 1024, total_m / 1024
        proc = _process_mem_mb()
        proc_txt = f"{proc:.0f} MB" if proc >= 0 else "-"
        mem_txt = f"{used_g:.1f} / {total_g:.1f} GB · 进程 {proc_txt}"   # 与任务管理器"内存"页一致
    else:
        mem_txt = "-"
    return (f'<span class="fv-sys-item">🖥 GPU：<b>{gpu}</b></span>'
            f'<span class="fv-sys-item">🎮 显存：<b>{vram}</b>{vram_bar}</span>'
            f'<span class="fv-sys-item">🧠 内存：<b>{mem_txt}</b></span>')


def render_logs(logs, current_stage: int = 99, done: bool = False) -> str:
    """logs: list[(stage:int, msg:str)] -> 彩色 HTML 日志
    颜色与顶部胶囊一致: 进行中=绿(fv-log-cur), 已完成=蓝(fv-log-done), 异常/停止=红(fv-log-0)
    current_stage=当前执行阶段; done=True 时全部已完成"""
    def _cls(s: int) -> str:
        if s == 0:
            return "fv-log-0"
        if done or (current_stage > 0 and s < current_stage):
            return "fv-log-done"
        return "fv-log-cur"
    return "".join(
        f'<div class="{_cls(s)}">[{s}/7] {_html.escape(m)}</div>'
        for s, m in logs)


def flow_html(stage: int, running: bool = False) -> str:
    """顶部流程徽章 (当前步高亮/已完成置绿; 仅胶囊形式, 无进度条)"""
    parts: list[str] = []
    for i, s in enumerate(FLOW_STEPS):
        idx = i + 1                       # 徽章序号对应 stage (1-6)
        if stage == 0:
            c = "fv-pend"
        elif idx < stage:
            c = "fv-done"
        elif idx == stage:
            c = "fv-cur"
        else:
            c = "fv-pend"
        parts.append(f'<span class="fv-step {c}">{s}</span>')
        if i < len(FLOW_STEPS) - 1:
            parts.append('<span class="fv-arrow">→</span>')
    return '<div class="fv-flow">' + "".join(parts) + "</div>"


def build_app() -> gr.Blocks:
    celebs = scan_celebrities()          # {label: {"mp3":.., "jpg":..}}
    # Gallery 数据: [(头像路径, 显示名)]; 头像缺失时用空占位
    celeb_entries = [(info["jpg"] or "", label) for label, info in celebs.items()]
    celeb_mp3_by_label = {label: info["mp3"] for label, info in celebs.items()}
    _first_mp3 = celeb_mp3_by_label.get(celeb_entries[0][1]) if celeb_entries else None

    def resolve_ref(mode: str, upload: str | None, celeb_path: str | None) -> str | None:
        if mode == "上传音频":
            return upload
        if mode == "音色库":
            return celeb_path
        return None

    with gr.Blocks(title="Fun-Voice 视频配音工作台") as demo:
        # 顶部: 品牌区 (左: 大标题+流程) + 环境/主题区 (右) — 细分隔线视觉分离
        # (强制深色 + 主题胶囊切换 JS 由 server.py launch(js=FORCE_DARK_JS) 全局注入)
        # GPU/显存/内存信息已移至日志上方监控条 (sys_stats_html + gr.Timer 实时刷新)
        _dk = "✓ 已配置" if (os.environ.get("VP_TRANSLATE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")) else "✗ 未配置(.env)"
        with gr.Row(elem_id="fv-topbar"):
            with gr.Column(scale=3, elem_id="fv-brand"):
                gr.Markdown("## Fun-Voice 视频配音工作台")
                gr.HTML(
                    '<div class="fv-flow">'
                    '<span class="fv-step">上传视频</span><span class="fv-arrow">→</span>'
                    '<span class="fv-step">人声分离</span><span class="fv-arrow">→</span>'
                    '<span class="fv-step">语音识别</span><span class="fv-arrow">→</span>'
                    '<span class="fv-step">原文翻译</span><span class="fv-arrow">→</span>'
                    '<span class="fv-step">克隆配音</span><span class="fv-arrow">→</span>'
                    '<span class="fv-step">混流出片</span>'
                    '</div>')
            with gr.Column(scale=1, elem_id="fv-env"):
                gr.Markdown(f"翻译服务：**DeepSeek API（{_dk}）**")
        state = gr.State(None)          # PipelineResult
        ref_state = gr.State(None)      # (ref_audio, ref_text, target_lang)
        job_ws = gr.State(None)         # 最近一次任务的 workspace (断点续跑用)
        running = gr.State(False)       # 任务运行中标记 (U3: 运行中禁用续跑/单句重跑)
        log_state = gr.State("")        # 当前日志 HTML (U6: 停止时追加"已停止")

        with gr.Tabs() as tabs:
            # ================= Tab1 输入参数 =================
            with gr.Tab("① 输入参数"):
                with gr.Row():
                    with gr.Column(scale=4):
                        video = gr.Video(label="源视频（本地上传）", sources=["upload"], height=200)
                        video_info_html = gr.Markdown("", visible=False)
                        ref_mode = gr.Radio(
                            ["上传音频", "音色库"],
                            value="上传音频", label="配音音色",
                            container=False, elem_id="fv-ref-mode")
                        ref_upload = gr.Audio(
                            label="上传参考音频（3-30 秒干净人声）",
                            sources=["upload"], type="filepath", visible=True,
                            elem_id="fv-ref-audio")
                        with gr.Column(visible=False) as celeb_panel:
                            celeb_gallery = gr.Gallery(
                                value=celeb_entries, columns=5, height=280,
                                label="音色库 · 点击头像可试听",
                                allow_preview=False, object_fit="cover",
                                selected_index=0 if celeb_entries else None)
                            celeb_audio = gr.Audio(
                                label="试听（当前选中音源）",
                                type="filepath", interactive=False, value=_first_mp3)
                        celeb_state = gr.State(_first_mp3)   # 当前选中音源的 mp3 路径
                        ref_text = gr.Textbox(
                            label="参考文本（可选，填写可提升克隆相似度）",
                            placeholder="该参考音频的逐字文本…",
                            elem_id="fv-ref-text")
                    with gr.Column(scale=5, elem_id="fv-params-group", variant="panel"):
                        # 参数顺序与流程步骤一致 (UI 修复轮 U7)
                        sep_model = gr.Radio(
                            ["MDX23C-8KFFT-InstVoc", "UVR-MDX onnx", "Kim_Vocal_2"],
                            value="MDX23C-8KFFT-InstVoc", label="人声分离")
                        overlap_slider = gr.Slider(
                            1, 16, value=4, step=1,
                            label="分离重叠 num_overlap",
                            info="重叠越大分离越平滑但越慢/占显存")
                        target_lang = gr.Radio(
                            [("中文", "Chinese"), ("English", "English"),
                             ("日语", "Japanese"), ("韩语", "Korean")],
                            value="Chinese", label="目标语言",
                            info="选择配音/翻译的目标语言")
                        batch_slider = gr.Slider(
                            1, 48, value=12, step=1,
                            label="TTS 批大小 (每批句数)",
                            info="每批句数（默认 12），推荐12-24：过小（接近逐句）或过大（约 60 句以上）易失败")
                        align_mode_dd = gr.Radio(
                            [("拉伸填充（默认）", "stretch"),
                             ("自然语速居中", "natural")],
                            value="stretch", label="配音时间轴对齐",
                            info="拉伸填充=变速填满字幕窗；自然语速居中=短句不变速、窗内中点对齐")
                        stretch_slider = gr.Slider(
                            0.6, 2.0, value=1.8, step=0.1,
                            label="配音拉伸上限",
                            info="中文比原文短时自动拉伸至原字幕窗长；压缩固定为 1/上限")
                        volume_slider = gr.Slider(
                            50, 200, value=100, step=5,
                            label="配音音量",
                            info="100 = 自动匹配原片响度；大于 100 再额外增强")
                        mix_bg_dd = gr.Radio(
                            [("不考虑背景音（默认）", "no"),
                             ("混合背景音", "yes")],
                            value="no", label="混流背景音",
                            info="混合背景音=把原视频伴奏以低音量混入成品 (分离不干净时可能有原人声残留)")
                        audio_stream_dd = gr.Dropdown(
                            choices=[], value=None, label="音频流（多音轨视频）",
                            info="上传视频后自动列出音轨, 选择要配音的一条")
                        quality_dd = gr.Radio(
                            [("无损直通（体积=源视频）", "copy"),
                             ("均衡压缩（体积约减 30-50%）", "balanced"),
                             ("体积优先（体积约减 50-70%）", "small")],
                            value="copy", label="成品画质")
                        with gr.Accordion("翻译 API 设置（可选，留空用 .env）", open=False):
                            trans_api_key = gr.Textbox(label="API Key", type="password",
                                                       placeholder="sk-...")
                            trans_base_url = gr.Textbox(label="Base URL",
                                                        placeholder="https://api.deepseek.com")
                            trans_model = gr.Textbox(label="模型名",
                                                     placeholder="deepseek-chat")
                        with gr.Accordion("ASR 识别设置（可选，提升专有名词识别）", open=False):
                            asr_hotwords = gr.Textbox(
                                label="热词（人名/地名/术语，逗号分隔）",
                                placeholder="例如: 华盛顿, 美联储, Biden")
                            asr_prompt = gr.Textbox(
                                label="提示词（上下文引导，可选）",
                                placeholder="例如: 以下是国际新闻播报内容。")
                        # 执行范围: 完整流程 / 仅到翻译 (完成至翻译, 改译文后可续跑继续配音)
                        stop_after_dd = gr.Radio(
                            [("完整流程（默认）", 0),
                             ("仅到翻译完成（先审译文）", 4)],
                            value=0, label="执行范围",
                            info="仅到翻译=不执行克隆配音/对齐/混流；在②字幕编辑改好译文后点「↻ 续跑上次」继续")
                        with gr.Row():
                            run_btn = gr.Button("▶ 开始配音", variant="primary")
                            stop_btn = gr.Button("■ 停止", elem_id="fv-stop-btn")
                            resume_btn = gr.Button("↻ 续跑上次", elem_id="fv-resume-btn")
                # 运行进度条 + 日志: 平铺于网页底部 (参考文本/开始配音/停止 共同下方, 整行宽)
                flow_box = gr.HTML(value=flow_html(0, False), elem_id="fv-flow-progress")
                sys_box = gr.HTML(value=sys_stats_html(), elem_id="fv-sys")
                sys_timer = gr.Timer(value=2)          # 每 2 秒刷新系统监控 (nvidia-smi 查询开销小, 1s 太频)
                sys_timer.tick(sys_stats_html, None, sys_box)
                log_box = gr.HTML(label="运行日志", value="", elem_id="fv-log")

            # ================= Tab2 字幕编辑 =================
            with gr.Tab("② 字幕编辑"):
                tab2_empty = gr.Markdown(
                    "👆 上传视频并点击 **开始配音**，字幕会出现在这里。\n\n"
                    "运行到「原文翻译」步骤后，即可在此浏览全文、逐句编辑译文。",
                    elem_id="fv-empty")
                gr.Markdown("##### 全文浏览（不带时间标签，点击复制图标可复制）")
                with gr.Row():
                    full_orig = gr.Textbox(label="原文全文", lines=10, interactive=False,
                                           max_lines=10, buttons=["copy"], elem_id="fv-full-orig")
                    full_zh = gr.Textbox(label="译文全文", lines=10, interactive=False,
                                         max_lines=10, buttons=["copy"], elem_id="fv-full-zh")
                gr.Markdown("##### 逐句编辑（选中一行在下方编辑；配音列 ✓=已完成 ⏳=合成中）")
                sub_df = gr.Dataframe(
                    headers=["#", "时间窗", "原文", "译文", "配音"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False, wrap=True)
                edit_panel = gr.Group(visible=False)
                with edit_panel:
                    edit_idx = gr.Number(label="选中句序号", value=0, visible=False)
                    edit_info = gr.Markdown("")
                    with gr.Row():
                        edit_orig = gr.Textbox(label="原文（只读）", interactive=False)
                        edit_zh = gr.Textbox(label="译文（可修改）", interactive=True, lines=3)
                    with gr.Row():
                        save_btn = gr.Button("💾 保存并重新合成此句", variant="primary")
                        edit_msg = gr.Markdown("")

            # ================= Tab3 结果下载 =================
            with gr.Tab("③ 结果下载"):
                gr.Markdown("👆 运行任务后，可在此**预览成品**并下载视频/音频/字幕；运行中结果会自动填充。",
                            elem_id="fv-empty")
                with gr.Row():
                    with gr.Column(scale=1):
                        out_video = gr.Video(label="生成视频预览")
                    with gr.Column(scale=1, elem_id="fv-dl-group", variant="panel"):
                        with gr.Row():
                            with gr.Column():
                                dl_video = gr.DownloadButton("⬇ 下载生成视频")
                            with gr.Column():
                                dl_audio = gr.DownloadButton("⬇ 下载配音音频")
                        with gr.Row():
                            with gr.Column():
                                dl_srt_orig = gr.DownloadButton("⬇ 下载原文字幕")
                            with gr.Column():
                                dl_srt_zh = gr.DownloadButton("⬇ 下载译文字幕")
                        info_html = gr.HTML("")
                gr.Markdown("##### 轨道试听（点击播放，对比分离与配音效果）")
                with gr.Row():
                    pl_source = gr.Audio(label="原音轨", type="filepath", interactive=False)
                    pl_vocals = gr.Audio(label="分离人声", type="filepath", interactive=False)
                    pl_inst = gr.Audio(label="分离伴奏", type="filepath", interactive=False)
                pl_dubbed = gr.Audio(label="成品配音", type="filepath", interactive=False)

        # ---------- 事件: 配音音色来源切换 ----------
        def _on_ref_mode(mode: str):
            return (gr.update(visible=mode == "上传音频"),
                    gr.update(visible=mode == "音色库"))
        ref_mode.change(_on_ref_mode, ref_mode, [ref_upload, celeb_panel])

        # ---------- 事件: 点击音色头像 -> 同步试听 + 记录选中 ----------
        def on_celeb_select(evt: gr.SelectData):
            try:
                idx = evt.index
                if isinstance(idx, (list, tuple)):
                    idx = idx[0]
                idx = int(idx)
                if celeb_entries and 0 <= idx < len(celeb_entries):
                    mp3 = celeb_mp3_by_label.get(celeb_entries[idx][1])
                    if mp3:
                        return gr.update(value=mp3), mp3
            except Exception:
                pass   # 点击非头像区域等, 静默忽略
            return gr.update(), None
        celeb_gallery.select(on_celeb_select, None, [celeb_audio, celeb_state])

        # ---------- 工具: 表格/输出构建 (供流式 yield 复用; 日志/进度条渲染函数见 build_app 头部) ----------
        def build_rows(res) -> list:
            if not res or not res.asr_segments:
                return []
            seg_dir = os.path.join(res.workspace, "03_zh")
            done_set = {int(f[:-4]) for f in os.listdir(seg_dir)
                        if f.endswith(".wav")} if os.path.isdir(seg_dir) else set()
            rows = []
            for i, (s, e, o) in enumerate(res.asr_segments):
                zh = res.zh_lines[i] if i < len(res.zh_lines) else ""
                if i in done_set:
                    status = "✓"
                elif zh.strip():
                    status = "⏳"
                else:
                    status = "-"
                rows.append([str(i + 1), f"{fmt_ts(s)} - {fmt_ts(e)}", o, zh, status])
            return rows

        def build_output(res, logs, done=False, stage=0, running=True):
            """返回 20 个 UI 输出 (流式中间态或最终态)
            顺序: log_box, sub_df, full_orig, full_zh, out_video, dl_video, dl_audio,
                  dl_srt_orig, dl_srt_zh, info_html, pl_source, pl_vocals, pl_inst,
                  pl_dubbed, tab2_empty, run_btn, flow_box, sys_box, edit_panel, edit_zh
            sys_box 随每帧刷新 (与 gr.Timer 双通道, 保证运行中显存/内存实时)
            done=False 时文件类组件(视频/音频/下载)不更新, 仅最终帧填充,
            避免生成器中途反复更新文件组件导致预览丢失;
            首帧 (res=None) 清空全部文件组件与编辑面板 (U1/U4), 按钮按 running 反馈 (M3)"""
            log_html = render_logs(logs, stage, done)
            run_upd = (gr.update(value="⏳ 运行中…" if running else "▶ 开始配音",
                                 interactive=not running))
            flow = flow_html(stage, running)
            sys_upd = gr.update(value=sys_stats_html())
            if not res:                return (log_html, [], "",
                        "", gr.update(value=None), gr.update(value=None),
                        gr.update(value=None), gr.update(value=None),
                        gr.update(value=None), "",
                        gr.update(value=None), gr.update(value=None),
                        gr.update(value=None), gr.update(value=None),
                        gr.update(visible=True), run_upd, flow, sys_upd,
                        gr.update(visible=False), gr.update(value=""))
            rows = build_rows(res)
            full_orig_text = "\n".join(o for _, _, o in res.asr_segments)
            full_zh_text = "\n".join(res.zh_lines)
            has_out = bool(res.output_video and os.path.exists(res.output_video))
            has_dub = bool(res.dubbed_audio and os.path.exists(res.dubbed_audio))
            has_srt = bool(res.workspace and os.path.exists(os.path.join(res.workspace, "03_zh.srt")))
            info = (info_card_html(res.drift_seconds, res.workspace, done,
                                   timings=res.timings)
                    if res.workspace else "")
            if done:                                   # 最终帧: 填充文件组件 + 解锁按钮
                return (log_html, rows,
                        full_orig_text, full_zh_text,
                        gr.update(value=res.output_video if has_out else None),
                        gr.update(value=res.output_video if has_out else None),
                        gr.update(value=res.dubbed_audio if has_dub else None),
                        gr.update(value=os.path.join(res.workspace, "03_orig.srt") if has_srt else None),
                        gr.update(value=os.path.join(res.workspace, "03_zh.srt") if has_srt else None),
                        info,
                        gr.update(value=res.source_audio if has_dub else None),
                        gr.update(value=res.vocals if has_dub else None),
                        gr.update(value=res.instrumental if has_dub else None),
                        gr.update(value=res.dubbed_audio if has_dub else None),
                        gr.update(visible=False), run_upd, flow, sys_upd,
                        gr.update(), gr.update())
            # 中间帧: 只更新文本/表格, 文件组件保持不变; 隐藏空状态; 保持按钮禁用
            return (log_html, rows,
                    full_orig_text, full_zh_text,
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), info,
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(visible=False), run_upd, flow, sys_upd,
                    gr.update(), gr.update())

        # ---------- 事件: 运行管道 (流式: 每阶段 yield 中间状态; 运行中锁死开始按钮) ----------
        def run_pipeline(video_path, mode, upload, celeb, rtext, sep, lang, stretch,
                         volume, quality, batch_size, align_mode, mix_bg,
                         audio_stream, overlap, trans_key, trans_url, trans_model,
                         asr_hot, asr_prompt, stop_after):
            if not video_path:
                raise gr.Error("请先上传源视频")
            ref_audio = resolve_ref(mode, upload, celeb)
            logs: list[tuple[int, str]] = []
            p = Pipeline()
            job_ws.value = p.workspace   # 记录任务 workspace, 供断点续跑
            state.value = None           # 新任务开始清空旧状态 (防 stop/失败后残留旧 res)
            ref_state.value = None
            running.value = True         # U3: 运行中锁定续跑/单句重跑
            res = None
            stage_now = 0
            audio_stream_idx = 0
            if audio_stream:
                try:
                    audio_stream_idx = int(str(audio_stream).split()[-1]) - 1
                except (ValueError, IndexError):
                    audio_stream_idx = 0
            translate_config = {}
            if trans_key:
                translate_config["api_key"] = trans_key
            if trans_url:
                translate_config["base_url"] = trans_url
            if trans_model:
                translate_config["model"] = trans_model
            asr_config = {}
            if asr_hot:
                asr_config["hotwords"] = asr_hot
            if asr_prompt:
                asr_config["initial_prompt"] = asr_prompt
            try:
                yield build_output(None, logs, stage=0, running=True)   # 首帧: 清空/复位 + 禁用按钮
                for stage, r, msg in p.run_iter(
                        video_path, ref_audio, rtext or None,
                        target_lang=lang,
                        stretch=(1.0 / max(stretch, 0.6), stretch),
                        separator=sep, volume_gain=volume / 100.0, quality=quality,
                        batch_size=batch_size, align_mode=align_mode,
                        mix_background=(mix_bg == "yes"),
                        audio_stream=audio_stream_idx, num_overlap=overlap,
                        translate_config=(translate_config or None),
                        asr_config=(asr_config or None),
                        stop_after=int(stop_after or 0) or None):
                    res = r
                    ref_state.value = (ref_audio, rtext or None, lang)  # U2: 每帧记录, 失败/停止后也可编辑单句
                    if msg:
                        logs.append((stage, msg))
                    stage_now = stage
                    sa = int(stop_after or 0) or None
                    done = stage == 7 or (sa == 4 and stage == 4 and msg.startswith("✅"))
                    log_state.value = render_logs(logs, stage, done)     # U6: 供停止按钮追加
                    yield build_output(res, logs, done=done, stage=stage,
                                       running=not done)
                state.value = res
                running.value = False
            except Exception as e:
                # 异常/停止时解锁开始按钮, 避免卡在禁用状态; 附带失败详情 (U2)
                logs.append((0, f"❌ 失败: {type(e).__name__}: {e}"))
                running.value = False
                try:
                    yield build_output(res, logs, done=False, stage=stage_now,
                                       running=False)
                except Exception:
                    pass
                raise

        run_event = run_btn.click(
            run_pipeline,
            [video, ref_mode, ref_upload, celeb_state, ref_text, sep_model,
             target_lang, stretch_slider, volume_slider, quality_dd, batch_slider,
             align_mode_dd, mix_bg_dd, audio_stream_dd, overlap_slider,
             trans_api_key, trans_base_url, trans_model, asr_hotwords, asr_prompt,
             stop_after_dd],
            [log_box, sub_df, full_orig, full_zh, out_video, dl_video,
             dl_audio, dl_srt_orig, dl_srt_zh, info_html,
             pl_source, pl_vocals, pl_inst, pl_dubbed, tab2_empty,
             run_btn, flow_box, sys_box, edit_panel, edit_zh])

        # ---------- 事件: 断点续跑 (复用上次任务 workspace, 自动跳过已完成阶段) ----------
        def _resume_ref(ws: str):
            """从 state.json 恢复 (ref_audio, ref_text, target_lang), 供编辑单句复用 (U2)"""
            try:
                import json as _json
                with open(os.path.join(ws, "state.json"), encoding="utf-8") as f:
                    params = _json.load(f)["params"]
                return (params.get("ref_audio"), params.get("ref_text"),
                        params.get("target_lang", "Chinese"))
            except Exception:
                return (None, None, "Chinese")

        def resume_pipeline(job_ws):
            if running.value:
                raise gr.Error("任务运行中，请等待完成")   # U3
            if not job_ws or not os.path.isfile(os.path.join(job_ws, "state.json")):
                raise gr.Error("没有可续跑的任务 (请先运行过「开始配音」)")
            logs: list[tuple[int, str]] = []
            p = Pipeline(workspace=job_ws)
            res = None
            stage_now = 0
            running.value = True
            try:
                yield build_output(None, logs, stage=0, running=True)   # 首帧: 清空/复位 + 禁用按钮
                for stage, r, msg in p.run_iter("", "", target_lang="Chinese"):
                    res = r
                    if msg:
                        logs.append((stage, msg))
                    stage_now = stage
                    done = stage == 7
                    log_state.value = render_logs(logs, stage, done)
                    yield build_output(res, logs, done=done, stage=stage,
                                       running=not done)
                state.value = res
                ref_state.value = _resume_ref(job_ws)      # U2: 恢复参考音色/语言
                running.value = False
            except Exception as e:
                # 异常/停止时解锁开始按钮, 避免卡在禁用状态; 附带失败详情 (U2)
                logs.append((0, f"❌ 续跑失败: {type(e).__name__}: {e}"))
                running.value = False
                try:
                    yield build_output(res, logs, done=False, stage=stage_now,
                                       running=False)
                except Exception:
                    pass
                raise

        resume_event = resume_btn.click(
            resume_pipeline,
            [job_ws],
            [log_box, sub_df, full_orig, full_zh, out_video, dl_video,
             dl_audio, dl_srt_orig, dl_srt_zh, info_html,
             pl_source, pl_vocals, pl_inst, pl_dubbed, tab2_empty,
             run_btn, flow_box, sys_box, edit_panel, edit_zh],
            cancels=[run_event])

        # 停止: 恢复按钮 + 日志追加"已停止" (U6); cancels 中断生成器, 模型释放由 pipeline finally 保证
        def on_stop(log_html):
            running.value = False
            return (gr.update(value=log_html + '<div class="fv-log-0">[stop] ⏹ 已停止</div>'),
                    gr.update(value="▶ 开始配音", interactive=True))
        stop_btn.click(on_stop, [log_state], [log_box, run_btn],
                       cancels=[run_event, resume_event])

        # ---------- 事件: 表格选中 -> 填充编辑面板 ----------
        def on_select(evt: gr.SelectData, df_val, st):
            if not st or df_val is None:
                return (gr.update(visible=False), "", "", "", 0)
            # df_val 是 pandas DataFrame (gradio type="pandas") — 不能用 not/len 布尔判断,
            # 也不能 list(df) 取行(那会得到列名); 统一转行列表
            if hasattr(df_val, "values"):
                data = [[str(c) for c in row] for row in df_val.values.tolist()]
            else:
                data = list(df_val)
            if not data:
                return (gr.update(visible=False), "", "", "", 0)
            row = evt.index[0]
            # 防御: gradio Dataframe select 行号可能含表头偏移 (表头行第一列非数字)
            if 0 <= row < len(data) and not str(data[row][0]).strip().isdigit():
                row -= 1
            if not 0 <= row < len(data):
                return (gr.update(visible=False), "", "", "", 0)
            idx = int(data[row][0])
            st_ts, en_ts = data[row][1].split(" - ")
            return (gr.update(visible=True),
                    f"**第 {idx} 句** · 时间窗 {st_ts} - {en_ts}",
                    data[row][2], data[row][3], idx)
        sub_df.select(on_select, [sub_df, state],
                      [edit_panel, edit_info, edit_orig, edit_zh, edit_idx])

        # ---------- 事件: 保存并重跑选中句 ----------
        def save_segment(idx, new_zh, st, ref):
            if running.value:
                raise gr.Error("任务运行中，请等待完成后再编辑单句")   # U3
            if not st or not idx:
                raise gr.Error("请先在表格中选择一句")
            res: PipelineResult = st
            p = Pipeline(workspace=res.workspace)
            ref_audio, rtext, lang = ref or (None, None, "Chinese")
            new_res, msg = p.rerun_segment(
                res, int(idx) - 1, new_zh, ref_audio, rtext, lang)
            state.value = new_res
            rows = build_rows(new_res)
            full_zh_text = "\n".join(new_res.zh_lines)
            return (rows, full_zh_text,
                    gr.update(value=new_res.output_video),
                    gr.update(value=new_res.output_video),
                    gr.update(value=new_res.dubbed_audio),
                    gr.update(value=os.path.join(new_res.workspace, "03_orig.srt")),
                    gr.update(value=os.path.join(new_res.workspace, "03_zh.srt")),
                    f"<div style='color:#7ad0a0'>✅ {msg}</div>",   # U8: 明确是单句重合成, 非整个任务完成
                    f"✅ {msg}",
                    gr.update(value=new_res.dubbed_audio))
        save_btn.click(save_segment,
                       [edit_idx, edit_zh, state, ref_state],
                       [sub_df, full_zh, out_video, dl_video, dl_audio, dl_srt_orig,
                        dl_srt_zh, info_html, edit_msg, pl_dubbed])

        # ---------- 事件: 上传视频后显示基本信息 + 音轨预检 (C1) + 音频流下拉 (C2) ----------
        def on_video_change(vp):
            if not vp:
                return gr.update(value=""), gr.update(choices=[], value=None)
            from app.audio import probe_audio_streams, probe_video_info
            try:
                info = probe_video_info(vp)   # U5: 解析异常容错
            except Exception:
                info = None
            if not info:
                return (gr.update(value="⚠️ 无法解析视频"),
                        gr.update(choices=[], value=None))
            text = (f"时长 **{int(info.get('duration', 0)//60)}分{int(info.get('duration', 0)%60)}秒** "
                    f"· 分辨率 {info.get('width')}x{info.get('height')} "
                    f"· 编码 {info.get('video_codec', '?')} "
                    f"· 体积 {info.get('size_mb', 0):.0f}MB")
            try:
                n = probe_audio_streams(vp)
            except RuntimeError as e:
                return (gr.update(value=f"⚠️ {e}"), gr.update(choices=[], value=None))
            if n == 0:
                return (gr.update(value=text + " · ⚠️ 无音轨"),
                        gr.update(choices=[], value=None, interactive=False))
            choices = [f"音轨 {i + 1}" for i in range(n)]
            return (gr.update(value=text + f" · 音轨 {n} 条"),
                    gr.update(choices=choices, value=choices[0], interactive=True))
        video.change(on_video_change, video, [video_info_html, audio_stream_dd])
        video.upload(on_video_change, video, [video_info_html, audio_stream_dd])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=True,
        theme=gr.themes.Base())   # 与 server.py DARK_THEME 一致 (深色)
