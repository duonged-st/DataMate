FROM nvidia/cuda:12.8.2-devel-ubuntu24.04 AS build
ARG LLAMA_VERSION=v1.0.0
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git cmake g++ make && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_VERSION}" https://github.com/ggml-org/llama.cpp.git /llama.cpp \
    && cmake -S /llama.cpp -B /build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120" \
      -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /build --target llama-server -j 4 \
    && cp /build/bin/llama-server /llama-server

FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libgomp1 libcurl4t64 && rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements.txt .
RUN python3 -m venv /opt/datamate-venv && /opt/datamate-venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/datamate-venv/bin:$PATH"
COPY --from=build /llama-server /usr/local/bin/llama-server
COPY models/Qwen3.5-9B-Q4_K_M.gguf models/Qwen3.5-9B-Q4_K_M.gguf
RUN echo "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8  models/Qwen3.5-9B-Q4_K_M.gguf" | sha256sum -c -
COPY datamate datamate
COPY training_code training_code
COPY verification verification
COPY confidence_policy.json README.md predict.sh serve.sh benchmark.py report.py threshold.py prepare_assets.py verify_outputs.py cpu_smoke.py ./
RUN cd /code/training_code && python3 -c 'import json, pathlib; p=pathlib.Path("data_manifest.json"); assert p.is_file() and json.loads(p.read_text())["train"]["questions"] == 3000; assert pathlib.Path("data/train.zip").is_file()' \
    && chmod +x /code/predict.sh /code/serve.sh
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT []
CMD ["bash", "/code/predict.sh"]
