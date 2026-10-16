"""On-disk format for Qwen3-TTS cloned voices.

A cloned voice is the `VoiceClonePromptItem` that Qwen3-TTS's Base model
derives from a reference clip: the clip's speech codes (`ref_code`), a speaker
embedding (`ref_spk_embedding`) and the clip's transcript. Cloning is
zero-shot, so this file is all there is to "the trained voice".

Stored as `<name>.safetensors` (tensors) with the scalar fields in the
safetensors string metadata - no pickle, so a voice file can't run code.

Identical copies live in deploy/dgx-spark/qwen3-tts/ (reads them) and
deploy/dgx-spark/qwen3-tts-voice-clone/ (writes them). Keep the two in sync.
"""

import re
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT_VERSION = "1"
VOICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def save_voice(path: str, item: Any, *, base_model: str, language: str, extra: dict[str, str]) -> None:
    tensors = {"ref_spk_embedding": item.ref_spk_embedding.detach().float().cpu().contiguous()}
    if item.ref_code is not None:
        tensors["ref_code"] = item.ref_code.detach().cpu().contiguous()
    metadata = {
        "format_version": FORMAT_VERSION,
        "base_model": base_model,
        "language": language,
        "x_vector_only_mode": str(bool(item.x_vector_only_mode)),
        "icl_mode": str(bool(item.icl_mode)),
        "ref_text": item.ref_text or "",
        **extra,
    }
    save_file(tensors, path, metadata=metadata)


def read_metadata(path: str) -> dict[str, str]:
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata() or {})


def load_voice(path: str, *, device: Any, dtype: torch.dtype) -> tuple[Any, dict[str, str]]:
    """Returns (VoiceClonePromptItem, metadata)."""
    from qwen_tts import VoiceClonePromptItem

    with safe_open(path, framework="pt") as f:
        metadata = dict(f.metadata() or {})
        keys = set(f.keys())
        embedding = f.get_tensor("ref_spk_embedding")
        ref_code = f.get_tensor("ref_code") if "ref_code" in keys else None
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported voice format {metadata.get('format_version')!r}")
    item = VoiceClonePromptItem(
        ref_code=ref_code.to(device) if ref_code is not None else None,
        ref_spk_embedding=embedding.to(device=device, dtype=dtype),
        x_vector_only_mode=metadata.get("x_vector_only_mode") == "True",
        icl_mode=metadata.get("icl_mode") == "True",
        ref_text=metadata.get("ref_text") or None,
    )
    return item, metadata
