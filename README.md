# Fun-Voice — Video Dubbing Workbench

> **English** | [中文](./README.zh-CN.md)

Upload a video → vocal separation → speech recognition → translation → voice-clone dubbing → timeline alignment → final mix.
Built for **Chinese dubbing of news / long videos**: the voice is **yours to choose** (upload a reference clip or pick from a voice library — e.g. anime-style or other custom voices). Supports **resume after interruption** and **concurrent-run mutual exclusion**.

## Features

- **7-step offline pipeline**: audio extraction → vocal separation (MDX23C / UVR-MDX onnx / Kim_Vocal_2) → ASR (faster-whisper large-v3-turbo, local & offline) → translation (OpenAI-compatible API + local opus-mt fallback, auto-degrades without an API key) → voice cloning (Qwen3-TTS-12Hz-1.7B, x-vector mode) → timeline alignment (librosa time-stretch without pitch shift, auto-absorbs EN↔ZH duration gaps) → final mix
- **Resume after interruption**: every step persists to disk with a `state.json`; after a crash/failure, click "↻ Resume last run" to skip completed stages (TTS re-synthesizes only missing lines)
- **Concurrent-run exclusion**: a global task lock (`workspace/.run.lock`) prevents GPU memory collisions; stale locks from crashes are taken over automatically (6h / dead-process detection)
- **Live streaming progress**: the Gradio UI refreshes logs / subtitle tables stage by stage; TTS batch progress and model-load hints; an info card shows **GPU memory usage and per-stage timings** in real time
- **Subtitle editing + per-line re-dubbing**: edit translated lines, re-synthesize a single line, re-align and re-mix (keeps loudness / quality / stretch / align-mode parameters)
- **Dual alignment modes**: "stretch-fill" (time-stretch to fill the subtitle window, default) and "natural pacing centered" (short lines keep natural speed, centered in window); over-long lines are padded with silence / clipped; tail drift is zeroed on long videos
- **Multi-track support**: audio tracks are listed automatically so you pick which one to dub; missing/corrupt audio reports a Chinese error before extraction
- **Background-mix option**: optionally mix the original instrumental at low volume (0.35) into the output to keep background music / studio ambience; off by default
- **Tunable ASR quality**: hotwords (names/places/terms, hard candidates for faster-whisper) and prompt injection for better news terminology accuracy (UI collapsible panel or `VP_ASR_HOTWORDS` / `VP_ASR_PROMPT`)
- **Configurable parameters**: separation model, separation overlap `num_overlap`, target language, stretch limit, align mode, volume gain, quality tier, TTS batch size, background-mix, translation API (Key/Base URL/Model via UI panel or .env), ASR hotwords/prompt — all adjustable in the UI

## Quick Start

```bash
# 1. Install dependencies (Python 3.12, uv recommended)
uv sync

# 2. Configure environment (copy .env.example to .env and fill in)
#    - Required: QWEN3_TTS_MODEL (local Qwen3-TTS model directory)
#    - Recommended: DEEPSEEK_API_KEY (translation; without it, falls back to local opus-mt, lower quality)
#    - Optional: VP_MODELS (default <parent>/models, holds ffmpeg + separation models)

# 3. Launch
Run `启动FunVoice.bat` (menu includes environment check: Python / GPU / models / ffmpeg / SoX).
```

### Model Preparation

| Model | Location | Notes |
|---|---|---|
| ffmpeg / ffprobe | `models/ffmpeg/bin/` | video/audio processing |
| MDX23C + UVR-MDX onnx + Kim_Vocal_2 | `models/MDX_Net_Models/` | vocal separation (recommend shipping with your deployment) |
| faster-whisper large-v3-turbo | `models/faster-whisper/` | ASR (auto-downloads if missing; speed up with `HF_ENDPOINT=https://hf-mirror.com`) |
| Qwen3-TTS-12Hz-1.7B-Base | set by `QWEN3_TTS_MODEL` | voice cloning (download locally, then configure the path) |

## Configuration (.env)

| Variable | Description | Default |
|---|---|---|
| `VP_MODELS` | model root directory | `<project>/models` |
| `FFMPEG_PATH` | ffmpeg executable path | `<VP_MODELS>/ffmpeg/bin/ffmpeg.exe` |
| `QWEN3_TTS_MODEL` | Qwen3-TTS model directory | — |
| `VP_ASR_LANG` | ASR source language (e.g. `en`, skips auto-detection) | auto |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | translation API | — |
| `VP_TRANSLATE_API_KEY` / `VP_TRANSLATE_BASE_URL` / `VP_TRANSLATE_MODEL` | translation API (OpenAI-compatible; takes priority over `DEEPSEEK_*`) | — |
| `VP_TRANSLATE_LOCAL_MODEL` | local fallback translation model | `Helsinki-NLP/opus-mt-en-zh` |
| `VP_REF_SECONDS` | max reference-voice seconds (truncate longer clips) | 12 |
| `VP_TTS_BATCH_SIZE` | lines per TTS batch | 12 |
| `VP_TTS_TEMPERATURE` / `VP_TTS_TOP_P` / `VP_TTS_TOP_K` / `VP_TTS_REPETITION_PENALTY` | TTS generation parameters (optional) | not passed |
| `HF_ENDPOINT` | HuggingFace mirror (e.g. `https://hf-mirror.com`) | official |

## Tests

```bash
cd fun-voice
PYTHONUTF8=1 ./.venv/Scripts/python.exe -m unittest discover -s tests
```

All 84 tests are fake/mock (no real models, no ffmpeg/network dependency) and finish in seconds. Coverage: separation pad boundary / GPU memory release / audio extraction / translation fallback chain / TTS validation & batching / pipeline yield sequence / **resume & concurrency lock / timeline alignment (incl. natural pacing) / mix parameters (faststart / background mix) / ASR hotwords & prompt passthrough**.

## Architecture

```
fun-voice/
├── app/
│   ├── server.py                # entry point (gradio launch, dark theme / footer injection)
│   ├── ui.py                    # Gradio UI (params panel / subtitle editing / result download / live monitor / resume / execution range)
│   ├── pipeline.py              # 7-step pipeline orchestration: generator yield + resume + global lock + stop_after
│   ├── audio.py                 # ffmpeg wrapper (extract / time-stretch / loudness / mix, lru_cache dedup)
│   └── engines/
│       ├── separator.py         # MDX23C (torch) / UVR-MDX onnx / Kim_Vocal_2
│       ├── asr.py               # faster-whisper (CTranslate2, hotwords / prompt configurable)
│       ├── translator.py        # OpenAI-compatible translation + opus-mt local fallback
│       └── tts.py               # Qwen3-TTS cloning (reference validation / batching / generation params)
├── tests/                       # 84-case regression suite (unittest, all fake/mock, seconds)
├── tools/
│   ├── adapt_gpu.py             # one-click GPU tier adaptation (cu128/cu126/cpu, rewrites pyproject sources)
│   └── fetch_models.py          # one-click missing-model download (whisper/Qwen3-TTS/ffmpeg; MDX is manual)
├── workspace/job-*/             # per-job intermediates + state.json (resume)
├── .github/workflows/ci.yml     # GitHub Actions CI (uv sync + 84 tests, windows-latest, CPU torch)
├── 启动FunVoice.bat             # one-click menu: start / check / repair / adapt GPU / fetch models
├── pyproject.toml + uv.lock     # dependencies (uv-managed, auto-downloads Python 3.12)
├── LICENSE / README.md (EN) / README.zh-CN.md (中文) / .env.example / .gitignore
└── (sibling dir) ../models/     # model dir: MDX/whisper/ffmpeg/voice library — not in the repo, see "Deploying on a new machine"
```

Pipeline yield sequence (full run): `[1,1,2,2,3,3,4,4,5,5,5,5,6,6,6,7,7,7]`
(each stage emits "start hint (with estimate) + done (actual time)"; stage 5 emits four: start / load / batch / done, batch count varies with batch count; with "translate only" execution range: `[1,1,2,2,3,3,4,4,4]`).

## Deploying on a New Machine

> The target machine needs internet access (models and dependencies are fetched online); for fully offline use, copy the whole directory (see below).

### Fast Path (recommended)

1. **Make sure `tools\uv\uv.exe` exists** — if not, download `uv-x86_64-pc-windows-msvc.zip` from [uv releases](https://github.com/astral-sh/uv/releases) and put it there (uv is a single-file exe, **it does not need Python itself**).
2. Double-click `启动FunVoice.bat` → **[4] Adapt GPU**: auto-detects your GPU and switches the torch tier in `pyproject.toml` to a matching build.
3. **[3] Repair environment (uv sync)**: auto-downloads Python 3.12 + all dependencies (**no system Python required** — uv manages its own).
4. Prepare models (see below).

### Python Fallbacks

- No system **Python** → no install needed, `uv sync` downloads an isolated 3.12.
- System Python version ≠ 3.12 → ignored, uv uses its own 3.12.
- `requires-python = ">=3.12,<3.13"`: only 3.12.x is supported.

### Different GPU → torch tier

torch 2.8.0 (Windows) has three tiers: **cu128 / cu126 / cpu**. Choose by the `CUDA Version` at the top of `nvidia-smi` (the highest CUDA your driver supports):

| Tier | Applies to | Menu [4] auto-switch |
|---|---|---|
| cu128 (default) | driver CUDA ≥ 12.8 | ✅ |
| cu126 | driver CUDA 12.6~12.7 | ✅ |
| cpu | no NVIDIA / driver too old | ✅ (slow; engines auto-fall back to CPU) |

No GPU or driver too old (CUDA < 12.6; torch 2.8 has no older tier) → pick cpu, or update the driver and use a cu tier. Manual switch: edit `[[tool.uv.index]]` url and `[tool.uv.sources]` index names in `pyproject.toml` (`pytorch-cu128` ↔ `pytorch-cu126` ↔ `pytorch-cpu`), then re-run `uv sync`.

### Model Preparation (corresponds to bat [2] environment check)

> ⚠️ **The model directory lives one level ABOVE the project folder** (sibling of `fun-voice`), not inside it. Full layout:

```
your-deploy-dir/
├── fun-voice/                  ← code pulled from GitHub (this repo)
│   ├── app/  tests/  tools/  ...
│   └── 启动FunVoice.bat
└── models/                     ← model dir (SIBLING of fun-voice! code resolves it via ../models)
    ├── MDX_Net_Models/         ← MDX 3 models (no official direct link, place manually)
    │   ├── MDX23C-8KFFT-InstVoc_HQ.ckpt      (428 MB)
    │   ├── UVR-MDX-NET-Inst_HQ_3.onnx        (64 MB)
    │   └── Kim_Vocal_2.onnx                  (64 MB)
    ├── faster-whisper/         ← [5] auto-download
    ├── ffmpeg/bin/             ← [5] auto-download
    └── ... (other model dirs)
```

| Model | Location | How to get |
|---|---|---|
| MDX separation (3 .ckpt/.onnx) | `models\MDX_Net_Models\` (**project parent**) | **place manually** (no official direct link, see sources below) |
| faster-whisper | `models\faster-whisper\large-v3-turbo\` | menu **[5]** auto-download (hf-mirror) |
| Qwen3-TTS | `K:\HuggingFace\models\Qwen3-TTS-12Hz-1.7B-Base\` | menu **[5]** auto-download (hf-mirror) |
| ffmpeg | `models\ffmpeg\bin\ffmpeg.exe` | menu **[5]** auto-download (gyan.dev) |

**MDX 3-model sources (official links; download and place into `models\MDX_Net_Models\` yourself)**:
- `MDX23C-8KFFT-InstVoc_HQ.ckpt` (MIT license) → https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md
- `UVR-MDX-NET-Inst_HQ_3.onnx` / `Kim_Vocal_2.onnx` (UVR community models) → https://ultimatevocalremover.com

After placing MDX models, run bat **[2] Environment check** to confirm `[OK] MDX23C vocal separator`.

### Celebrity Voice Library (local only, not in repo)

> ⚠️ This project does **not** bundle or distribute any celebrity voice material (portrait/voice rights and AI-clone compliance — please prepare licensed material yourself).

Voice library layout (sibling dir `models\celebrities30s\` above the project):

```
models/
└── celebrities30s/
    ├── celebrities30s.json5          # voice config (label / image / audio mapping)
    ├── Chinese/                      # each voice = 1 cover jpg + 1 30s reference mp3
    │   ├── your-voice-name.jpg
    │   └── your-voice-name.mp3
    ├── English/
    ├── Japanese/
    └── Korean/
```

- Material suggestions: **your own voice**, friends with permission, AI-generated voices, or licensed libraries (avoid real celebrities)
- If the layout is wrong, the UI voice panel shows empty (other features unaffected; fill it in anytime)

### Fully Offline

Copy the whole directory: project root (including `.venv`, `tools\uv\`, `models\`, `K:\HuggingFace\models\`). `.venv` holds pip-installed wheels (no absolute-path dependencies), so `启动FunVoice.bat` works right away with no internet.

## Known Issues & Pitfalls

- **TTS batch size**: default 12 is a measured compromise — too large a batch (67-line scale) stalls on long decode sequences; too small (near per-line) also stalls on consecutive `generate` calls. On an RTX 3060 12GB, 24 measured ~22% faster than 12 (peak 8.98GB/12GB); adjustable in the UI.
- **flash-attn warning**: "flash-attn is not installed" at startup comes from a module-level print in the qwen_tts 25Hz legacy path; **the 12Hz flow is unaffected** (it uses torch SDPA — no flash-attn needed).
- **SoX warning**: also from a qwen_tts x-vector side path; not executed in the current flow, safe to ignore; installing SoX and adding it to PATH removes it.
- **Translation without key**: auto-falls back to local opus-mt (English→Chinese only, ~300MB first-time download); quality is better with an API key.
- **Concurrency limit**: only one task at a time (global lock); stale locks from crashes are taken over automatically (no manual deletion).

## Acceptance Checklist

| Milestone | Content | Status |
|---|---|---|
| M1 Environment | uv env + engine smoke tests | ✅ |
| M2 Engines | separation / ASR / translation / TTS independently usable | ✅ |
| M3 Pipeline | full pipeline + timeline alignment | ✅ (fake-engine tests cover it; run a real sample yourself) |
| M4 UI | Gradio + progress + subtitle editing + params panel | ✅ |
| M5 Hardening | GPU memory management / E2E tests / resume / concurrency lock | ✅ (84 tests) |
