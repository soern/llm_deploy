"""Import stub for nagisa - installed only when the real package is missing.

qwen_asr's __init__ eagerly imports its forced aligner, which does
`import nagisa` (a DyNet-based Japanese tokenizer with no aarch64 wheel).
This deploy never loads the forced aligner (no timestamps), so the import
only has to succeed. Any actual use raises.
"""
__version__ = "0.0.0+nixclaw-stub"


def __getattr__(name):
    raise ImportError(
        f"nagisa is stubbed in this image (nagisa.{name} unavailable); "
        "the Qwen3 forced aligner is not supported here"
    )
