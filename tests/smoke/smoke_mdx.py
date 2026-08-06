"""M1 冒烟测试: MDX 人声分离模型验证

两个目标:
  1. MDX23C-8KFFT-InstVoc_HQ.ckpt (torch 权重) — 验证可加载 + 输出网络结构(供 M2 实现推理)
  2. UVR-MDX-NET-Inst_HQ_3.onnx (onnxruntime 兜底) — 验证可加载 + hash 匹配配置

用法:
    python tests/smoke/smoke_mdx.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MDX_DIR = os.path.join(os.environ.get("VP_MODELS", r"K:\视频翻译与配音\models"), "MDX_Net_Models")
CKPT_PATH = os.path.join(MDX_DIR, "MDX23C-8KFFT-InstVoc_HQ.ckpt")
ONNX_PATH = os.path.join(MDX_DIR, "UVR-MDX-NET-Inst_HQ_3.onnx")
YAML_PATH = os.path.join(MDX_DIR, "model_data", "mdx_c_configs", "model_2_stem_full_band_8k.yaml")
REG_PATH = os.path.join(MDX_DIR, "model_data", "model_data.json")


def md5_tail(path: str) -> str:
    """UVR 约定: 取文件末尾 10000KB 的 md5 作为模型指纹"""
    with open(path, "rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
            return hashlib.md5(f.read()).hexdigest()
        except OSError:
            return hashlib.md5(open(path, "rb").read()).hexdigest()


def main() -> int:
    import torch
    import yaml

    # ---------- 1. MDX23C torch 权重 ----------
    print(f"=== MDX23C-8KFFT-InstVoc_HQ.ckpt ===")
    if not os.path.exists(CKPT_PATH):
        print(f"[FAIL] 找不到 ckpt: {CKPT_PATH}")
        return 1
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    keys = list(sd.keys())
    print(f"[OK] ckpt 加载成功, state_dict {len(keys)} 个张量")
    for k in keys[:30]:
        v = sd[k]
        print(f"    {k:48s} {tuple(v.shape)}")
    print(f"    ... (共 {len(keys)} 项)")

    cfg = yaml.safe_load(open(YAML_PATH, encoding="utf-8"))
    audio_cfg, model_cfg = cfg["audio"], cfg["model"]
    print(f"[OK] yaml 配置: n_fft={audio_cfg['n_fft']} hop={audio_cfg['hop_length']} "
          f"dim_f={audio_cfg['dim_f']} dim_t={audio_cfg['dim_t']} sr={audio_cfg['sample_rate']}")
    print(f"    网络: channels={model_cfg['num_channels']} growth={model_cfg['growth']} "
          f"scales={model_cfg['num_scales']} blocks/scale={model_cfg['num_blocks_per_scale']} "
          f"subbands={model_cfg['num_subbands']} act={model_cfg['act']} norm={model_cfg['norm']}")

    # ---------- 2. onnx 兜底 ----------
    print(f"\n=== UVR-MDX-NET-Inst_HQ_3.onnx ===")
    if not os.path.exists(ONNX_PATH):
        print("[FAIL] 找不到 onnx")
        return 1
    reg = json.load(open(REG_PATH, encoding="utf-8"))
    h = md5_tail(ONNX_PATH)
    params = reg.get(h)
    if params:
        print(f"[OK] hash 匹配配置: dim_f={params['mdx_dim_f_set']} dim_t={params['mdx_dim_t_set']} "
              f"n_fft={params['mdx_n_fft_scale_set']} stem={params['primary_stem']} comp={params['compensate']}")
    else:
        print(f"[WARN] hash 未在 model_data.json 中找到: {h}")
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"[OK] onnx 加载成功: input '{inp.name}' {inp.shape}")

    print("\n[PASS] MDX 冒烟完成: ckpt 权重可读, yaml 配置可解析, onnx 可加载")
    return 0


if __name__ == "__main__":
    sys.exit(main())
