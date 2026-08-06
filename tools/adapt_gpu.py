#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""适配本机 GPU -> torch 安装档位 (cu128 / cu126 / cpu)

换机部署时运行: 检测 nvidia-smi 驱动支持的 CUDA 版本, 自动改写
pyproject.toml 的 torch/torchaudio 下载源 ([[tool.uv.index]] + [tool.uv.sources]),
然后提示运行 `uv sync` 重新安装依赖。

用法:
    python tools/adapt_gpu.py                 # 自动检测并改写 (改前自动备份 pyproject.toml.bak)
    python tools/adapt_gpu.py --check         # 只检测本机应使用的档位, 不改写
    python tools/adapt_gpu.py --pyproject X   # 指定 pyproject 路径 (测试用)

档位规则 (torch 2.8.0 Windows):
    cu128  NVIDIA + 驱动支持 CUDA >= 12.8
    cu126  NVIDIA + 驱动支持 CUDA 12.6 ~ 12.7
    cpu    无 NVIDIA GPU, 或驱动过老 (<12.6, 建议同时更新驱动)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(PROJECT_ROOT, "pyproject.toml")
TIERS = ("cu128", "cu126", "cpu")


# ---------------- 检测 ----------------
def detect_cuda_version() -> str | None:
    """nvidia-smi 输出的驱动 CUDA Version (驱动支持的最高 CUDA runtime 版本)"""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True,
                             text=True, timeout=10)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
        return f"{m.group(1)}.{m.group(2)}" if m else None
    except Exception:
        return None


def choose_tier(cuda_ver: str | None) -> str:
    """无 NVIDIA -> cpu; >=12.8 -> cu128; >=12.6 -> cu126; 更老 -> cpu (建议更新驱动)"""
    if cuda_ver is None:
        return "cpu"
    major, minor = map(int, cuda_ver.split("."))
    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    return "cpu"          # 驱动过老: torch 2.8 无更老 CUDA 档, 只能 cpu 或更新驱动


# ---------------- pyproject 读写 ----------------
def current_tier(pyproject: str = PYPROJECT) -> str | None:
    """读取 pyproject.toml 当前配置的档位 (从 index url 或 sources index 名)"""
    try:
        text = open(pyproject, encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"download\.pytorch\.org/whl/([a-z0-9]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"pytorch-(cu\d+|cpu)", text)
    return m.group(1) if m else None


def rewrite_tier(tier: str, pyproject: str = PYPROJECT) -> str:
    """把 pyproject.toml 的 torch 档位改为 tier (cu128/cu126/cpu). 返回改写后全文"""
    if tier not in TIERS:
        raise ValueError(f"非法档位: {tier}, 可选 {TIERS}")
    text = open(pyproject, encoding="utf-8").read()
    # 1) index url 档位:  .../whl/cu128  ->  .../whl/{tier}
    text = re.sub(r"(download\.pytorch\.org/whl/)[a-z0-9]+", rf"\g<1>{tier}", text)
    # 2) index name 与 sources index:  pytorch-cu128 / pytorch-cpu -> pytorch-{tier}
    text = re.sub(r"pytorch-(?:cu\d+|cpu)", f"pytorch-{tier}", text)
    return text


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    pyproject = PYPROJECT
    if "--pyproject" in sys.argv:
        pyproject = sys.argv[sys.argv.index("--pyproject") + 1]

    cuda_ver = detect_cuda_version()
    tier = choose_tier(cuda_ver)
    cur = current_tier(pyproject)

    print("=" * 52)
    print("  GPU / torch 档位适配")
    print("=" * 52)
    if cuda_ver:
        print(f"  检测到 NVIDIA GPU (nvidia-smi), 驱动 CUDA Version: {cuda_ver}")
        print(f"  -> 推荐档位: {tier}")
        if tier == "cpu" and (tuple(map(int, cuda_ver.split("."))) < (12, 6)):
            print("  ⚠️  驱动 CUDA < 12.6: torch 2.8 无更老 CUDA 档,")
            print("      建议更新显卡驱动; 也可先用 cpu 档(慢但可跑)")
    else:
        print("  未检测到 NVIDIA GPU (或 nvidia-smi 不可用)")
        print("  -> 推荐档位: cpu (无 GPU 加速, 各引擎自动回退 CPU)")
    print(f"  当前 pyproject 档位: {cur or '(未识别)'}")

    if check_only:
        print(f"  [CHECK] 本机应使用: {tier} (未改写)")
        return 0
    if tier == cur:
        print(f"  [OK] 已是 {tier} 档, 无需修改")
        return 0

    backup = pyproject + ".bak"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(pyproject, backup)
        print(f"  [BACKUP] 已备份原配置 -> {os.path.basename(backup)}")
    new_text = rewrite_tier(tier, pyproject)
    with open(pyproject, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    print(f"  [DONE] pyproject.toml 已切换为 {tier} 档")
    print()
    print("  下一步: 运行 uv sync 重新安装依赖:")
    print("      tools\\uv\\uv.exe sync")
    print("  或回到 FunVoice 菜单选 [3] Repair environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
