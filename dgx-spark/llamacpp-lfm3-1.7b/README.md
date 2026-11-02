# llama.cpp deployment for LiquidAI/LFM3-1.7B (NVIDIA DGX Spark)

Fallback to `../sglang-lfm3-1.7b-nvfp4` (the NVFP4 default): GGUF **Q8_0** on
the stock multi-arch `ghcr.io/ggml-org/llama.cpp:server-cuda13` image
(linux/arm64 included, checked via manifest), no build step. Needs llama.cpp
>= v0.8.0 (the lfm3 architecture); the floating tag is newer. Port **8010**
(same as the NVFP4 deploy and `../qwen3-tts`; `start.sh` refuses to start if
the port is bound; override with `PORT=`).

- `MODEL_FILE=LFM3-1.7B-Q4_K_M.gguf` for the 4-bit GGUF (llmtest: equal quality,
  ~1.6x the generation speed on CPU).
- 131072 ctx, 4 slots (32K each), f16 KV (~1.5GB). Sampling defaults from the
  model card.
- **Unverified on the Spark.** Unknown: whether the CI image ships GB10
  (sm_121) kernels. `start.sh` warns if no layers were offloaded to the GPU;
  then build llama.cpp with `-DCMAKE_CUDA_ARCHITECTURES=121a` as in
  `../../jetson-orin-nano/lfm3-1.7b/Dockerfile` (change arch 87 -> 121a).
- llama.cpp also has an `NVFP4` ggml type, but there is no official LFM3-1.7B
  NVFP4 GGUF; the NVFP4 safetensors need SGLang.
