"""Import stub for torchaudio - installed only when the real package is missing.

qwen_tts eagerly imports its 25Hz tokenizer, whose speech_vq.py does
`import torchaudio.compliance.kaldi`. The 12Hz models this deploy serves
never call it, and NVIDIA's GB10 PyTorch image may not ship a torchaudio
built against its torch. Any actual use raises.
"""
__version__ = "0.0.0+nixclaw-stub"
