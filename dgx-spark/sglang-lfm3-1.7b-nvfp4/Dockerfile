# SGLang + the (still unmerged) LFM3 support patch from sgl-project/sglang#46858.
# Upstream images cannot serve LiquidAI/LFM3-1.7B-NVFP4 yet: no `Lfm3ForCausalLM`
# in SGLang main or vLLM. start.sh downloads the PR diff once into
# ./lfm3-sglang-pr.diff (so the build stays pinned after first run) and this
# Dockerfile overlays its python/ part onto the base image's sglang package.
# Drop this file and use the stock image once the PR is released.
ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13
FROM ${BASE_IMAGE}

COPY lfm3-sglang-pr.diff filter_pr_diff.py /tmp/

RUN set -eux; \
    SITE="$(python3 -c 'import os,sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"; \
    python3 /tmp/filter_pr_diff.py /tmp/lfm3-sglang-pr.diff /tmp/lfm3-python.diff; \
    cd "$SITE"; \
    git apply -p2 --check /tmp/lfm3-python.diff; \
    git apply -p2 /tmp/lfm3-python.diff; \
    python3 -c "import sglang.srt.models.lfm3 as m; print('lfm3 model module OK:', m.__file__)"
