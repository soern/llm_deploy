# llama.cpp `llama-server` built for Jetson Orin (Ampere, sm_87) on the
# JetPack 7 / L4T R39 CUDA 13 SBSA userspace. See README.md.
#
# Built on the Jetson itself by start.sh. CUDA_VERSION must not exceed the
# host driver's CUDA version (`nvidia-smi` header; 13.2 on jetson3).

ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG LLAMA_CPP_REF=master
ARG CUDA_ARCHITECTURES=87
# nvcc on the flash-attention template instances needs ~1GB+ per job; the
# Orin Nano's 8GB unified RAM OOMs with -j$(nproc).
ARG BUILD_JOBS=3

RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout "${LLAMA_CPP_REF}" \
    && git rev-parse HEAD > /src/LLAMA_CPP_COMMIT

# --allow-shlib-undefined: libcuda.so comes from the host driver at runtime
# (mounted by nvidia-container-runtime), not from the build image.
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-server -j "${BUILD_JOBS}"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/LLAMA_CPP_COMMIT /LLAMA_CPP_COMMIT

ENTRYPOINT ["/usr/local/bin/llama-server"]
