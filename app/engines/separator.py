"""人声分离引擎: MDX23C (torch) 主实现 + ONNX 兜底

MDX23C-8KFFT-InstVoc_HQ.ckpt 网络结构 (基于 ZFTurbo/music-source-separation-training
models/mdx23c_tfc_tdf_v3.py 的官方实现重写, MIT 许可):
    - 输入: 时域波形 (B, 2, chunk) -> STFT (B, 4, 4096, T) [2声道x2实虚]
    - cac2cws: 4 subbands 拆到通道 -> (B, 16, 1024, T)
    - first_conv: 1x1 (16 -> 128); 之后 transpose(-1,-2) 在 (B,C,T,F) 布局上跑
    - encoder: 5 scales (TFC_TDF + Downscale 2x2 stride2), 通道 128->768, 频率 1024->32
    - bottleneck: TFC_TDF(768)
    - decoder: 5 scales (Upscale + skip 拼接 + TFC_TDF)
    - x * first_conv_out (reduce artifacts) -> final_conv(cat(mix, x)) -> 32 通道 (2 target x 16)
    - cws2cac -> (B, 2target, 4, 4096, T) -> iSTFT -> 时域 (B, 2target, 2, chunk)
    - TFC_TDF 块: x = tfc2(tfc1(x) + tdf(x)) + shortcut(x); tdf 为 F->F/4->F 频率维线性
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()   # 保证 .env 中 VP_MODELS 生效 (与 audio.py 同款)

# 默认模型路径 (相对项目根定位, 可通过 VP_MODELS 环境变量覆盖)
DEFAULT_MDX_DIR = os.path.join(
    os.environ.get("VP_MODELS", str(Path(__file__).resolve().parents[3] / "models")),
    "MDX_Net_Models",
)
MDX23C_CKPT = os.path.join(DEFAULT_MDX_DIR, "MDX23C-8KFFT-InstVoc_HQ.ckpt")

# ---- MDX23C 配置: 单一事实来源 = model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml ----
# 启动时读取; 文件缺失/损坏时回退内置默认值 (与 yaml 一致的抄录, 仅作兜底)
MDX23C_DEFAULTS = dict(
    n_fft=8192, hop_length=1024, dim_f=4096, dim_t=256,
    sample_rate=44100, num_channels=2, num_subbands=4, num_channels_model=128,
    growth=128, num_scales=5, num_blocks_per_scale=2, bottleneck_factor=4,
    norm="InstanceNorm", act="gelu",
    num_overlap=4, min_mean_abs=0.001,
)
MDX23C_YAML = os.path.join(DEFAULT_MDX_DIR, "model_data", "mdx_c_configs",
                           "model_2_stem_full_band_8k.yaml")
# yaml (audio/model/inference 段) 键 -> MDX23C_CFG 键
_MDX23C_YAML_MAP = {
    ("audio", "n_fft"): "n_fft",
    ("audio", "hop_length"): "hop_length",
    ("audio", "dim_f"): "dim_f",
    ("audio", "dim_t"): "dim_t",
    ("audio", "sample_rate"): "sample_rate",
    ("audio", "num_channels"): "num_channels",
    ("audio", "min_mean_abs"): "min_mean_abs",
    ("model", "num_channels"): "num_channels_model",
    ("model", "num_subbands"): "num_subbands",
    ("model", "growth"): "growth",
    ("model", "num_scales"): "num_scales",
    ("model", "num_blocks_per_scale"): "num_blocks_per_scale",
    ("model", "bottleneck_factor"): "bottleneck_factor",
    ("model", "norm"): "norm",
    ("model", "act"): "act",
    ("inference", "num_overlap"): "num_overlap",
}


def _load_mdx23c_cfg() -> dict:
    """读 8k.yaml 构建 MDX23C 配置; 失败回退内置默认 (保持启动不崩)"""
    try:
        import yaml
        with open(MDX23C_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = dict(MDX23C_DEFAULTS)
        for (sec, key), dst in _MDX23C_YAML_MAP.items():
            if isinstance(data.get(sec), dict) and key in data[sec]:
                cfg[dst] = data[sec][key]
        return cfg
    except Exception as e:
        print(f"[separator] 读取 MDX23C 配置失败, 回退内置默认: {e}")
        return dict(MDX23C_DEFAULTS)


MDX23C_CFG = _load_mdx23c_cfg()


def get_norm(norm_type: str):
    if norm_type == "BatchNorm":
        return nn.BatchNorm2d
    if norm_type == "InstanceNorm":
        return nn.InstanceNorm2d
    return nn.Identity


def get_act(act_type: str):
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown act: {act_type}")


class _NormActConv(nn.Module):
    """Downscale/Upscale 的 conv 容器 (key: .conv.0/.2)"""

    def __init__(self, c_in: int, c_out: int, scale: int, transposed: bool):
        super().__init__()
        norm = get_norm("InstanceNorm")(c_in, affine=True)
        act = get_act("gelu")
        conv = (nn.ConvTranspose2d if transposed else nn.Conv2d)(
            c_in, c_out, kernel_size=scale, stride=scale, bias=False
        )
        self.conv = nn.Sequential(norm, act, conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TFC_TDF(nn.Module):
    """单个 tfc_tdf 块: x = tfc2(tfc1(x) + tdf(x)) + shortcut(x)
    在 (B, C, T, F) 布局上运行 (F 为最后维, Linear 直接作用)"""

    def __init__(self, in_c: int, c: int, f: int, bn: int):
        super().__init__()
        norm, act = get_norm("InstanceNorm"), get_act("gelu")
        self.tfc1 = nn.Sequential(norm(in_c, affine=True), act, nn.Conv2d(in_c, c, 3, 1, 1, bias=False))
        self.tdf = nn.Sequential(
            norm(c, affine=True), act,
            nn.Linear(f, f // bn, bias=False),
            norm(c, affine=True), act,
            nn.Linear(f // bn, f, bias=False),
        )
        self.tfc2 = nn.Sequential(norm(c, affine=True), act, nn.Conv2d(c, c, 3, 1, 1, bias=False))
        self.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shortcut(x)
        x = self.tfc1(x)
        x = x + self.tdf(x)
        x = self.tfc2(x)
        return x + s


class TFC_TDF_Stack(nn.Module):
    """多个 tfc_tdf 块 (key: .blocks.N), 首个块输入 in_c 可不同于 c"""

    def __init__(self, in_c: int, c: int, f: int, bn: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(TFC_TDF(in_c, c, f, bn))
            in_c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class MDX23Net(nn.Module):
    """MDX23C ConvTDFNet (官方 TFC_TDF_net 实现)"""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        c = dict(MDX23C_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        n, l = c["num_scales"], c["num_blocks_per_scale"]
        cm, g, bn = c["num_channels_model"], c["growth"], c["bottleneck_factor"]
        scale = 2
        self.num_subbands = c["num_subbands"]
        self.num_target_instruments = 2                     # Vocals + Instrumental
        dim_c = self.num_subbands * c["num_channels"] * 2   # 16

        f = c["dim_f"] // self.num_subbands                 # 1024

        self.first_conv = nn.Conv2d(dim_c, cm, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for _ in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF_Stack(cm, cm, f, bn, l)
            block.downscale = _NormActConv(cm, cm + g, scale, transposed=False)
            f //= scale
            cm += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF_Stack(cm, cm, f, bn, l)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(n):
            block = nn.Module()
            block.upscale = _NormActConv(cm, cm - g, scale, transposed=True)
            f *= scale
            cm -= g
            block.tfc_tdf = TFC_TDF_Stack(2 * cm, cm, f, bn, l)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(cm + dim_c, cm, 1, 1, 0, bias=False),
            get_act("gelu"),
            nn.Conv2d(cm, self.num_target_instruments * dim_c, 1, 1, 0, bias=False),
        )

    def cac2cws(self, x: torch.Tensor) -> torch.Tensor:
        k = self.num_subbands
        b, c, f, t = x.shape
        return x.reshape(b, c, k, f // k, t).reshape(b, c * k, f // k, t)

    def cws2cac(self, x: torch.Tensor) -> torch.Tensor:
        k = self.num_subbands
        b, c, f, t = x.shape
        return x.reshape(b, c // k, k, f, t).reshape(b, c // k, f * k, t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_audio, chunk) -> (B, 2target, C_audio, chunk)
        stft = _stft(x, self.cfg)
        mix = x = _cac2cws_audio(stft, self.num_subbands)
        first_conv_out = x = self.first_conv(x)
        x = x.transpose(-1, -2)                      # (B, C, T, F)

        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)
        x = self.bottleneck_block(x)
        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)                      # (B, C, F, T)
        x = x * first_conv_out
        x = self.final_conv(torch.cat([mix, x], 1))  # (B, 2target*16, 1024, T)
        x = self.cws2cac(x)                          # (B, 2target*4, 4096, T)
        b, c, f, t = x.shape
        x = x.reshape(b, self.num_target_instruments, -1, f, t)  # (B, 2, 4, 4096, T)
        return _istft_audio(x, self.cfg)             # (B, 2, 2, chunk)


def _stft(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """wave (B, C, L) -> (B, C*2, dim_f, T) [C 声道 x 2 实虚]"""
    n_fft, hop, dim_f = cfg["n_fft"], cfg["hop_length"], cfg["dim_f"]
    window = torch.hann_window(n_fft, periodic=True).to(x.device)
    b, c, _ = x.shape
    spec = torch.stft(x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=hop,
                      window=window, center=True, return_complex=True)
    spec = torch.view_as_real(spec).permute(0, 3, 1, 2)      # (B*C, 2, bins, T)
    spec = spec.reshape(b, c * 2, spec.shape[-2], spec.shape[-1])
    return spec[..., :dim_f, :]


def _cac2cws_audio(spec: torch.Tensor, k: int) -> torch.Tensor:
    b, c, f, t = spec.shape
    return spec.reshape(b, c, k, f // k, t).reshape(b, c * k, f // k, t)


def _istft_audio(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """x: (B, target, 4, dim_f, T) -> (B, target, 2, chunk)"""
    n_fft, hop, dim_f = cfg["n_fft"], cfg["hop_length"], cfg["dim_f"]
    window = torch.hann_window(n_fft, periodic=True).to(x.device)
    n = n_fft // 2 + 1
    b, tgt, _, _, t = x.shape
    x = torch.cat([x, torch.zeros(b, tgt, x.shape[-3], n - dim_f, t, device=x.device)], -2)  # (B,tgt,4,4097,T)
    x = x.reshape(b, tgt, 2, 2, n, t).reshape(-1, 2, n, t)    # 4 = 2ch x 2实虚
    x = x.permute(0, 2, 3, 1)
    x = x[..., 0] + x[..., 1] * 1j
    out = torch.istft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
    return out.reshape(b, tgt, 2, -1)


def load_mdx23c(ckpt_path: str = MDX23C_CKPT, device: str = "cuda") -> MDX23Net:
    """加载 MDX23C 权重, strict 校验"""
    model = MDX23Net()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"MDX23C 权重不匹配: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return model.to(device).eval()


class MDX23Separator:
    """MDX23C 人声分离器: 模型内部完成 STFT->分离->iSTFT, 输出双声道时域"""

    def __init__(self, ckpt_path: str = MDX23C_CKPT, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.cfg = MDX23C_CFG
        self.sample_rate = self.cfg["sample_rate"]
        self.chunk_size = self.cfg["hop_length"] * (self.cfg["dim_t"] - 1)   # 261120
        self.trim = self.cfg["n_fft"] // 2
        self.model = load_mdx23c(ckpt_path, device)

    def separate(self, wave: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """分离人声/伴奏. wave: (C, L) float32. 返回 (vocals, instrumental) 各 (C, L)

        num_overlap (cfg): 重叠推理次数 — 相邻窗口重叠平均, 消除分块边界伪影
        min_mean_abs (cfg): 静音块跳过阈值 (mean(|x|) 低于则直接输出静音, 省算力)"""
        if sr != self.sample_rate:
            import librosa
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sample_rate)
        wave_t = torch.from_numpy(np.ascontiguousarray(wave)).float().to(self.device)
        if wave_t.ndim == 1:
            wave_t = wave_t.unsqueeze(0).repeat(2, 1)

        n = wave_t.shape[1]
        if n == 0:
            empty = np.zeros((2, 0), dtype=np.float32)
            return empty, empty
        gen = self.chunk_size - 2 * self.trim
        pad = (gen - n % gen) % gen   # n 恰为 gen 整数倍时 pad=0, 避免多跑一整块
        num_overlap = int(self.cfg.get("num_overlap", 1))
        min_mean_abs = float(self.cfg.get("min_mean_abs", 0.0))
        step = max(gen // num_overlap, 1)
        out_len = n + pad
        wave_p = torch.cat([torch.zeros(2, self.trim, device=self.device), wave_t,
                            torch.zeros(2, pad + self.trim, device=self.device)], dim=1)

        acc_v = torch.zeros(2, out_len, device=self.device)
        acc_i = torch.zeros(2, out_len, device=self.device)
        cnt = torch.zeros(out_len, device=self.device)
        body = slice(self.trim, self.chunk_size - self.trim)
        with torch.no_grad():
            for i in range(0, out_len - gen + 1, step):
                chunk = wave_p[:, i:i + self.chunk_size]
                if min_mean_abs > 0 and float(torch.mean(torch.abs(chunk))) < min_mean_abs:
                    continue    # 静音块: 不推理不累积, 输出保持静音
                out = self.model(chunk.unsqueeze(0))          # (1, 2, 2, chunk)
                acc_v[:, i:i + gen] += out[0, 0, :, body]
                acc_i[:, i:i + gen] += out[0, 1, :, body]
                cnt[i:i + gen] += 1

        cnt = cnt.clamp(min=1)
        vocals = (acc_v / cnt)[:, :n].cpu().numpy()
        inst = (acc_i / cnt)[:, :n].cpu().numpy()
        return vocals, inst


# ==================== UVR-MDX onnx 分离器 ====================

MDX_MODELS_DIR = os.path.join(
    os.environ.get("VP_MODELS", r"K:\视频翻译与配音\models"), "MDX_Net_Models")

# ---- UVR-MDX onnx 模型注册表: 参数单一事实来源 = model_data/model_data.json ----
# json 以文件哈希为键 (UVR 生成, 非标准 MD5/SHA256, 无法运行时自算); 哈希作为稳定标识常量
_ONNX_MODEL_HASHES = {
    "UVR-MDX-NET-Inst_HQ_3": "55657dd70583b0fedfba5f67df11d711",
    "Kim_Vocal_2": "970b3f9492014d18fefeedfe4773cb42",
}
MODEL_DATA_JSON = os.path.join(MDX_MODELS_DIR, "model_data", "model_data.json")
# 兜底默认 (与 json 一致的手工抄录, 仅在读取失败时使用)
_ONNX_DEFAULTS = {
    "UVR-MDX-NET-Inst_HQ_3": {
        "path": os.path.join(MDX_MODELS_DIR, "UVR-MDX-NET-Inst_HQ_3.onnx"),
        "dim_f": 3072, "dim_t": 256, "n_fft": 6144,
        "compensate": 1.022, "stem": "Instrumental",   # 主输出为伴奏, 人声=原-伴奏
        "num_overlap": 4, "min_mean_abs": 0.001,        # 同 MDX23C: 重叠推理 + 静音跳过
    },
    "Kim_Vocal_2": {
        "path": os.path.join(MDX_MODELS_DIR, "Kim_Vocal_2.onnx"),
        "dim_f": 3072, "dim_t": 256, "n_fft": 7680,
        "compensate": 1.009, "stem": "Vocals",          # 主输出为人声
        "num_overlap": 4, "min_mean_abs": 0.001,
    },
}


def _load_onnx_models() -> dict:
    """读 model_data.json, 按哈希键取 onnx 参数 (mdx_dim_t_set 为 2 的指数); 失败回退默认"""
    try:
        with open(MODEL_DATA_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[separator] 读取 model_data.json 失败, 回退内置默认: {e}")
        return {k: dict(v) for k, v in _ONNX_DEFAULTS.items()}
    out = {}
    for name, h in _ONNX_MODEL_HASHES.items():
        base = dict(_ONNX_DEFAULTS[name])
        entry = data.get(h)
        if entry:
            if "mdx_dim_f_set" in entry:
                base["dim_f"] = entry["mdx_dim_f_set"]
            if "mdx_dim_t_set" in entry:
                base["dim_t"] = 2 ** entry["mdx_dim_t_set"]   # 8 -> 256
            if "mdx_n_fft_scale_set" in entry:
                base["n_fft"] = entry["mdx_n_fft_scale_set"]
            if "compensate" in entry:
                base["compensate"] = entry["compensate"]
            if "primary_stem" in entry:
                base["stem"] = entry["primary_stem"]
        out[name] = base
    return out


ONNX_MODELS = _load_onnx_models()


class MDXOnnxSeparator:
    """UVR-MDX onnx 人声分离器 (onnxruntime).

    架构: STFT 4通道复谱 → onnx 网络(输出分离谱) → iSTFT 恢复时域.
    onnx 输出为 primary_stem 茎; 另一茎 = 原信号 - primary_stem * compensate.
    接口与 MDX23Separator 一致: separate(wave, sr) -> (vocals, instrumental)
    """

    def __init__(self, model_name: str = "UVR-MDX-NET-Inst_HQ_3"):
        import onnxruntime as ort
        cfg = ONNX_MODELS[model_name]
        self.cfg = cfg
        self.dim_f, self.dim_t = cfg["dim_f"], cfg["dim_t"]
        self.n_fft, self.hop = cfg["n_fft"], 1024
        self.compensate = cfg["compensate"]
        self.stem = cfg["stem"]
        self.sample_rate = 44100
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = self.hop * (self.dim_t - 1)
        self.trim = self.n_fft // 2

        path = cfg["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"onnx 模型不存在: {path}")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if torch.cuda.is_available() else ["CPUExecutionProvider"])
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 3
        self.sess = ort.InferenceSession(path, sess_options=sess_opt, providers=providers)
        # 预加载 warmup
        self.sess.run(None, {"input": np.zeros((1, 4, self.dim_f, self.dim_t), np.float32)})

    # ---- STFT / ISTFT (torch, 布局与 MDX23C 一致) ----
    def _stft(self, x: np.ndarray) -> np.ndarray:
        """x: (2, chunk) float32 -> (1, 4, dim_f, T) numpy"""
        xt = torch.from_numpy(x).float()
        xt = xt.reshape(-1, self.chunk_size)
        X = torch.stft(xt, n_fft=self.n_fft, hop_length=self.hop,
                       window=torch.hann_window(self.n_fft, periodic=True),
                       center=True, return_complex=True)
        X = torch.view_as_real(X).permute(0, 3, 1, 2)          # (B,2,bins,T)
        X = X.reshape(-1, 2, 2, self.n_bins, self.dim_t).reshape(-1, 4, self.n_bins, self.dim_t)
        return X[:, :, :self.dim_f].numpy()

    def _istft(self, spec: np.ndarray) -> np.ndarray:
        """spec: (1, 4, dim_f, T) -> (2, chunk) numpy"""
        x = torch.from_numpy(spec).float()
        b, _, _, t = x.shape
        freq_pad = torch.zeros(b, 4, self.n_bins - self.dim_f, t)
        x = torch.cat([x, freq_pad], dim=-2)                    # (B,4,n_bins,T)
        x = x.reshape(-1, 2, 2, self.n_bins, t).reshape(-1, 2, self.n_bins, t)
        x = x.permute(0, 2, 3, 1).contiguous()                  # (B*2, T, bins, 2)
        x = torch.view_as_complex(x)                            # (B*2, T, bins)
        w = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop,
                        window=torch.hann_window(self.n_fft, periodic=True),
                        center=True)                            # (B*2, chunk)
        return w.numpy()                                        # 单 chunk(B=1) -> (2, chunk)

    def separate(self, wave: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """分离人声/伴奏. wave: (C, L) float32. 返回 (vocals, instrumental) 各 (C, L)

        num_overlap (cfg): 重叠推理次数 — 相邻窗口重叠平均, 消除分块边界伪影
        min_mean_abs (cfg): 静音块跳过阈值 (mean(|x|) 低于则直接输出静音, 省算力)"""
        if sr != self.sample_rate:
            import librosa
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sample_rate)
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        if wave.ndim == 1:
            wave = np.stack([wave, wave])

        n = wave.shape[1]
        if n == 0:
            empty = np.zeros((2, 0), dtype=np.float32)
            return empty, empty

        # 归一化到峰值 (MDX onnx 对归一化输入更稳定), 处理后恢复
        peak = float(np.max(np.abs(wave)))
        if peak > 1e-6:
            wave = wave / peak

        gen = self.chunk_size - 2 * self.trim
        pad = (gen - n % gen) % gen   # n 恰为 gen 整数倍时 pad=0, 避免多跑一整块
        num_overlap = int(self.cfg.get("num_overlap", 1))
        min_mean_abs = float(self.cfg.get("min_mean_abs", 0.0))
        step = max(gen // num_overlap, 1)
        out_len = n + pad
        wave_p = np.concatenate([np.zeros((2, self.trim), np.float32), wave,
                                 np.zeros((2, pad + self.trim), np.float32)], axis=1)

        acc_p = np.zeros((2, out_len), np.float32)
        cnt = np.zeros(out_len, np.float32)
        body = slice(self.trim, self.chunk_size - self.trim)
        for i in range(0, out_len - gen + 1, step):
            chunk = wave_p[:, i:i + self.chunk_size]
            if min_mean_abs > 0 and float(np.mean(np.abs(chunk))) < min_mean_abs:
                continue    # 静音块: 不推理不累积, 输出保持静音
            spec = self._stft(chunk)                          # (1,4,dim_f,T)
            out = self.sess.run(None, {"input": spec})[0]     # (1,4,dim_f,T)
            wav = self._istft(out)                            # (2, chunk)
            acc_p[:, i:i + gen] += wav[:, body]
            cnt[i:i + gen] += 1

        cnt = np.clip(cnt, 1, None)
        primary = (acc_p / cnt)[:, :n] * peak

        # 主茎与另一茎
        if self.stem == "Vocals":
            vocals, inst = primary, wave * peak - primary * self.compensate
        else:  # Instrumental
            inst, vocals = primary, wave * peak - primary * self.compensate
        return vocals.astype(np.float32), inst.astype(np.float32)


def get_separator(kind: str = "mdx23c"):
    """分离器工厂:
    'mdx23c'/'MDX23C-8KFFT-InstVoc' -> MDX23C torch
    'mdx-onnx'/'UVR-MDX onnx'/'UVR-MDX-NET-Inst_HQ_3' -> UVR-MDX onnx (主茎伴奏)
    'Kim_Vocal_2' -> Kim_Vocal_2 onnx (主茎人声)"""
    if kind in ("mdx-onnx", "UVR-MDX onnx", "UVR-MDX-NET-Inst_HQ_3"):
        return MDXOnnxSeparator("UVR-MDX-NET-Inst_HQ_3")
    if kind == "Kim_Vocal_2":
        return MDXOnnxSeparator("Kim_Vocal_2")
    return MDX23Separator()
