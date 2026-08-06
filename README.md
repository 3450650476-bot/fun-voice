# Fun-Voice 视频配音工作台

上传视频 → 人声分离 → 语音识别 → 翻译 → 声音克隆配音 → 时间轴对齐 → 混流出片。
面向**新闻/长视频中文配音**场景：音色由用户指定（上传音频或音色库，如二次元/其他声音），支持断点续跑与并发互斥。

## 功能特性

- **7 步离线管道**：音轨提取 → 人声分离（MDX23C / UVR-MDX onnx / Kim_Vocal_2）→ ASR（faster-whisper large-v3-turbo，本地离线）→ 翻译（OpenAI 兼容 API + 本地 opus-mt 兜底，无 key 自动降级）→ 声音克隆（Qwen3-TTS-12Hz-1.7B，x-vector 模式）→ 时间轴对齐（librosa 变速不变调，自动吸收中英文时长差）→ 混流
- **断点续跑**：每步产物落盘 + `state.json` 状态持久化；中断/失败后点「↻ 续跑上次」自动跳过已完成阶段（TTS 逐句补缺，不重算已完成句）
- **并发互斥**：全局任务锁（`workspace/.run.lock`），防多任务显存互踩；崩溃残留锁自动接管（6 小时/进程消失判定）
- **流式进度**：Gradio 界面逐阶段刷新日志/字幕表格；TTS 批量进度与模型加载提示；信息卡片实时显示 **GPU 显存占用与各阶段耗时**
- **字幕编辑 + 逐句重配音**：编辑译文后单句重合成并重新对齐混流（沿用响度/画质/变速/对齐模式参数）
- **时间轴对齐双模式**：「拉伸填充」（变速填满字幕窗，默认）与「自然语速居中」（短句不变速、窗内中点对齐）；超限句自动补静音/裁剪，长视频尾部漂移归零
- **多音轨支持**：上传后自动列出音轨下拉选择要配音的一条；无音轨/损坏视频在提取前给出中文报错
- **混流背景音**：可选把原视频伴奏以低音量（0.35）混入成品，保留背景音乐/演播室音效；默认不混合
- **ASR 识别质量可调**：热词（人名/地名/术语，faster-whisper 硬性候选词）与提示词注入，新闻专有名词识别准确率提升（UI 折叠面板或 `VP_ASR_HOTWORDS`/`VP_ASR_PROMPT`）
- **参数可配**：分离模型、分离重叠 num_overlap、目标语言、变速上限、对齐模式、音量增益、画质档位、TTS 批大小、混流背景音、翻译 API（Key/Base URL/模型，UI 面板或 .env）、ASR 热词/提示词均可在 UI 调整

## 快速开始

```bash
# 1. 安装依赖 (Python 3.12, 推荐 uv)
uv sync

# 2. 配置环境 (复制 .env.example 为 .env 并填写)
#    - 必填: QWEN3_TTS_MODEL (本地 Qwen3-TTS 模型目录)
#    - 推荐: DEEPSEEK_API_KEY (翻译; 不填则自动降级本地 opus-mt, 质量较低)
#    - 可选: VP_MODELS (默认 K:\视频翻译与配音\models, 含 ffmpeg 与分离模型)

# 3. 启动
启动 `启动FunVoice.bat`（菜单含环境检查，可检测 Python/GPU/模型/ffmpeg/SoX）。
```

### 模型准备

| 模型 | 位置 | 说明 |
|---|---|---|
| ffmpeg / ffprobe | `models/ffmpeg/bin/` | 视频/音频处理 |
| MDX23C + UVR-MDX onnx + Kim_Vocal_2 | `models/MDX_Net_Models/` | 人声分离（建议随项目打包） |
| faster-whisper large-v3-turbo | `models/faster-whisper/` | ASR（目标机无则自动从网络下载，可用 `HF_ENDPOINT=https://hf-mirror.com` 加速） |
| Qwen3-TTS-12Hz-1.7B-Base | 由 `QWEN3_TTS_MODEL` 指定 | 声音克隆（本地下载后配置路径） |

## 配置项（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `VP_MODELS` | 模型根目录 | `<项目>/models` |
| `FFMPEG_PATH` | ffmpeg 可执行文件路径 | `<VP_MODELS>/ffmpeg/bin/ffmpeg.exe` |
| `QWEN3_TTS_MODEL` | Qwen3-TTS 模型目录 | — |
| `VP_ASR_LANG` | ASR 源语言（如 `en`，免自动检测） | 自动 |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 翻译 API | — |
| `VP_TRANSLATE_API_KEY` / `VP_TRANSLATE_BASE_URL` / `VP_TRANSLATE_MODEL` | 翻译 API（OpenAI 兼容，优先级高于 DEEPSEEK_*） | — |
| `VP_TRANSLATE_LOCAL_MODEL` | 本地兜底翻译模型 | `Helsinki-NLP/opus-mt-en-zh` |
| `VP_REF_SECONDS` | 参考音色最长秒数（超长截取） | 12 |
| `VP_TTS_BATCH_SIZE` | TTS 每批句数 | 12 |
| `VP_TTS_TEMPERATURE` / `VP_TTS_TOP_P` / `VP_TTS_TOP_K` / `VP_TTS_REPETITION_PENALTY` | TTS 生成参数（可选） | 不传 |
| `HF_ENDPOINT` | HuggingFace 镜像（如 `https://hf-mirror.com`） | 官方 |

## 测试

```bash
cd fun-voice
PYTHONUTF8=1 ./.venv/Scripts/python.exe -m unittest discover -s tests
```

69 个用例全部 fake/mock（不加载真实模型、不依赖 ffmpeg/网络），秒级完成。覆盖：分离 pad 边界 / 显存释放 / 音轨提取 / 翻译降级链 / TTS 校验与批大小 / 管道 yield 序列 / **断点续跑与并发互斥 / 时间轴对齐（含自然语速居中）/ 混流参数（faststart/背景音混合）/ ASR 热词与提示词透传**。

## 架构

```
fun-voice/
├── app/
│   ├── server.py                # 启动入口 (gradio launch, 深色主题/页脚隐藏注入)
│   ├── ui.py                    # Gradio 界面 (参数面板/字幕编辑/结果下载/实时监控/续跑/执行范围)
│   ├── pipeline.py              # 7 步管道编排: 生成器 yield + 断点续跑 + 全局锁 + stop_after
│   ├── audio.py                 # ffmpeg 封装 (提取/变速/响度/混流, lru_cache 去重)
│   └── engines/
│       ├── separator.py         # MDX23C (torch) / UVR-MDX onnx / Kim_Vocal_2
│       ├── asr.py               # faster-whisper (CTranslate2, 热词/提示词可配)
│       ├── translator.py        # OpenAI 兼容翻译 + opus-mt 本地兜底
│       └── tts.py               # Qwen3-TTS 克隆 (参考校验/分批/生成参数可配)
├── tests/                       # 84 用例回归套件 (unittest, 全 fake/mock 秒级)
├── tools/
│   ├── adapt_gpu.py             # GPU 档位一键适配 (cu128/cu126/cpu, 改 pyproject 源)
│   └── fetch_models.py          # 缺失模型一键下载 (whisper/Qwen3-TTS/ffmpeg; MDX 需手动)
├── workspace/job-*/             # 每任务中间产物 + state.json (断点续跑)
├── .github/workflows/ci.yml     # GitHub Actions CI (uv sync + 84 用例)
├── 启动FunVoice.bat             # 一键菜单: 启动/检查/修复/适配GPU/下载模型
├── pyproject.toml + uv.lock     # 依赖 (uv 管理, 自动下载 Python 3.12)
├── LICENSE / README.md / .env.example
```

管道 yield 序列（正常全量）：`[1,1,2,2,3,3,4,4,5,5,5,5,6,6,6,7,7,7]`
（每阶段"开始提示(含预估) + 完成(实际用时)"；阶段 5 四条：开始/加载/批量/完成，批量消息次数随批数变化；"仅到翻译"执行范围时 `[1,1,2,2,3,3,4,4,4]`）。

## 换机部署

> 目标机器需能联网（模型与依赖在线获取）；完全离线时请整目录拷贝（见下）。

### 快速路径（推荐）
1. **确保 `tools\uv\uv.exe` 存在**——没有则从 [uv releases](https://github.com/astral-sh/uv/releases) 下载 `uv-x86_64-pc-windows-msvc.zip` 放入（uv 是单文件 exe，**本身不依赖 Python**）。
2. 双击 `启动FunVoice.bat` → **[4] Adapt GPU**：自动检测本机显卡，把 `pyproject.toml` 的 torch 档位切到匹配的版本。
3. **[3] Repair environment（uv sync）**：自动下载 Python 3.12 + 全部依赖（**系统无需预装 Python**，uv 自带管理）。
4. 准备模型（见下）。

### Python 兜底
- 系统**没有 Python** → 无需安装，`uv sync` 自动下载独立 3.12。
- 系统 Python 版本 ≠ 3.12 → 忽略，uv 用自己的 3.12。
- `requires-python = ">=3.12,<3.13"`：仅 3.12.x 受支持。

### GPU 不同 → torch 档位
torch 2.8.0（Windows）三档：**cu128 / cu126 / cpu**。按 `nvidia-smi` 顶部的 `CUDA Version`（驱动支持的最高 CUDA）选择：

| 档位 | 适用 | 菜单 [4] 自动切换 |
|---|---|---|
| cu128（默认） | 驱动 CUDA ≥ 12.8 | ✅ |
| cu126 | 驱动 CUDA 12.6~12.7 | ✅ |
| cpu | 无 NVIDIA / 驱动过老 | ✅（慢，各引擎自动回退 CPU） |

无 GPU 或驱动过老（CUDA < 12.6，torch 2.8 无更老档）时选 cpu，或更新显卡驱动后用 cu 档。手动切换：改 `pyproject.toml` 的 `[[tool.uv.index]]` url 与 `[tool.uv.sources]` index 名（`pytorch-cu128` ↔ `pytorch-cu126` ↔ `pytorch-cpu`）后重跑 `uv sync`。

### 模型准备（bat [2] 环境检查对应）

> ⚠️ **模型目录在项目文件夹的上一级**（与 fun-voice 同级），不是项目内部。完整布局：

```
你的部署目录/
├── fun-voice/                  ← 从 GitHub 拉下来的代码 (本仓库)
│   ├── app/  tests/  tools/  ...
│   └── 启动FunVoice.bat
└── models/                     ← 模型目录 (与 fun-voice 同级! 代码通过 ../models 定位)
    ├── MDX_Net_Models/         ← MDX 三模型 (官方无直链, 需手动放置)
    │   ├── MDX23C-8KFFT-InstVoc_HQ.ckpt      (428 MB)
    │   ├── UVR-MDX-NET-Inst_HQ_3.onnx        (64 MB)
    │   └── Kim_Vocal_2.onnx                  (64 MB)
    ├── faster-whisper/         ← [5] 自动下载
    ├── ffmpeg/bin/             ← [5] 自动下载
    └── ... (其他模型目录)
```

| 模型 | 位置 | 获取方式 |
|---|---|---|
| MDX 分离（3 个 .ckpt/.onnx） | `models\MDX_Net_Models\`（**项目上一级**） | **手动放置**（官方无直链）：从发布页 Release 附件 / 网盘下载后放进来 |
| faster-whisper | `models\faster-whisper\large-v3-turbo\` | 菜单 **[5]** 自动下载（hf-mirror） |
| Qwen3-TTS | `K:\HuggingFace\models\Qwen3-TTS-12Hz-1.7B-Base\` | 菜单 **[5]** 自动下载（hf-mirror） |
| ffmpeg | `models\ffmpeg\bin\ffmpeg.exe` | 菜单 **[5]** 自动下载（gyan.dev） |

放置好 MDX 后运行 bat **[2] Environment check** 确认 `[OK] MDX23C vocal separator`。

### 完全离线
整目录拷贝：项目根（含 `.venv`、`tools\uv\`、`models\`、`K:\HuggingFace\models\`）。`.venv` 是 pip 安装的 wheel（无绝对路径依赖），拷过去可直接用 `启动FunVoice.bat`，无需联网。

## 已知问题与踩坑

- **TTS 批大小**：默认 12 是实测折中——单批过大（67 句级）解码序列过长会卡死，过小（接近逐句）连续 `generate` 也会卡死。RTX 3060 12GB 实测 24 比 12 快约 22%（峰值显存 8.98GB/12GB），UI 可调。
- **flash-attn 警告**：启动时 "flash-attn is not installed" 来自 qwen_tts 25Hz 旧路径的模块级打印，**12Hz 流程不受影响**（当前实际使用 torch SDPA，无需安装 flash-attn）。
- **SoX 警告**：同样来自 qwen_tts 的 x-vector 旁路，当前流程不执行，可忽略；安装 SoX 并加入 PATH 可消除。
- **翻译无 key**：自动降级本地 opus-mt（仅支持英文→中文，首次自动下载 ~300MB），有 API key 时质量更好。
- **并发限制**：同一时刻只允许一个任务（全局锁）；崩溃后残留锁会自动接管（无需手动删除）。

## 验收对照

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 环境 | uv 环境 + 引擎冒烟 | ✅ |
| M2 引擎 | 分离/ASR/翻译/TTS 独立可用 | ✅ |
| M3 管道 | pipeline 全链路 + 时间轴对齐 | ✅（fake-engine 测试覆盖；真实样例建议自跑） |
| M4 UI | Gradio + 进度 + 字幕编辑 + 参数面板 | ✅ |
| M5 加固 | 显存管理 / E2E 测试 / 断点续跑 / 并发互斥 | ✅（84 用例） |
