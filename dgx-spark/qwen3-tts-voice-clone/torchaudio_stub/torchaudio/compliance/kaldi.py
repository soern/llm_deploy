def __getattr__(name):
    raise RuntimeError(
        f"torchaudio.compliance.kaldi.{name}: torchaudio is a stub in this image "
        "(only Qwen3-TTS 25Hz tokenizer models need it; use a 12Hz model)"
    )
