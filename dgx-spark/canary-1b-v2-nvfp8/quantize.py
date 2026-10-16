"""NVFP8 quantization for canary-1b-v2 (CanaryForConditionalGeneration).

Strategy
--------
- Load the base model in BF16 on CPU.
- Quantize every ``nn.Linear`` weight to FP8 e4m3 (``float8_e4m3fn``) with
  per-1x128 block FP32 scales (the NVFP8 / NVFP4-adjacent recipe used on
  Blackwell): scale = absmax of each 128-wide block / FP8_E4M3_MAX,
  q = clamp(w / scale).to(e4m3fn).
- Conv layers, embeddings, LayerNorm stay BF16 (small share of params).
- Quantized tensors are saved as a plain dict of tensors to
  ``QUANT_CACHE_DIR`` (no custom classes in the pickle → robust across
  transformers versions).
- At inference time the server re-applies the quantized tensors to a fresh
  base-model instance (see ``apply_quantized`` / ``make_linear_from_tensors``
  in server.py) and runs GEMMs in BF16 by dequantizing each layer's weight
  on the fly. Memory savings come from FP8 storage; speed stays at BF16
  parity. A torch._scaled_mm FP8 GEMM fast path can be layered on later.

Approximate footprint:
    base model FP32 on disk      : ~3.9 GB
    base model BF16 in RAM/VRAM  : ~1.9 GB
    quantized cache on disk      : ~0.75 GB  (FP8 weights + FP32 block scales)
    inference peak VRAM (b=1)    : ~1.5-2.5 GB
"""

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn

LOG = logging.getLogger("canary-quant")

BLOCK = 128
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FP8_MIN = -FP8_MAX


def quantize_linear_weight(w: torch.Tensor):
    """Quantize a Linear weight (out_features, in_features) to NVFP8.

    Returns (qweight uint8, scale fp32 [out, in//BLOCK]).
    """
    assert w.dim() == 2
    out_f, in_f = w.shape
    assert in_f % BLOCK == 0, f"in_features {in_f} not divisible by {BLOCK}"

    w = w.float()
    blocks = w.view(out_f, in_f // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / FP8_MAX
    q = (blocks / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = q.to(torch.float8_e4m3fn).view(out_f, in_f)

    return qweight.view(torch.uint8), scale.squeeze(2).to(torch.float32)


def _iter_linear_modules(model: nn.Module):
    """Yield (fqn, module) for every Linear that holds a 2-D ``weight``."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and isinstance(getattr(mod, "weight", None), torch.Tensor):
            fqn = name.replace(".weight", "").replace(".bias", "")
            yield fqn, mod


def build_quantized_state(model: nn.Module) -> dict:
    """Quantize all Linear layers of a model, return a saveable tensor dict."""
    quant = {}
    total, quantized = 0, 0
    for fqn, mod in _iter_linear_modules(model):
        w = mod.weight.detach()
        q, s = quantize_linear_weight(w)
        quant[fqn + ".qweight"] = q
        quant[fqn + ".scale"] = s
        b = getattr(mod, "bias", None)
        if isinstance(b, torch.Tensor):
            quant[fqn + ".bias"] = b.detach().cpu()
        quantized += w.numel()
        total += w.numel()
    LOG.info(
        "Quantized %d Linear layers: %d / %d params -> FP8 e4m3 + per-1x128 FP32 scales",
        len(quant) // 2, quantized, total,
    )
    return quant


def save_quantized(quant: dict, model_id: str, cache_dir: str | None = None) -> Path:
    cache_dir = Path(cache_dir or os.environ.get("QUANT_CACHE_DIR", "/models/canary-1b-v2-nvfp8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_id = model_id.replace("/", "__")
    out = cache_dir / f"{safe_id}.nvfp8.pt"
    torch.save({"block": BLOCK, "quant": quant}, out, _use_new_zipfile_serialization=True)
    LOG.info("Saved NVFP8 cache: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def load_quantized(path: Path) -> dict:
    """Return the full checkpoint: {'block': int, 'quant': tensor-dict}."""
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("block") != BLOCK or "quant" not in ckpt:
        raise ValueError(
            f"NVFP8 cache {path} malformed: expected keys 'block' (={BLOCK}) and 'quant'"
        )
    return ckpt
