# RunPod Serverless worker for neddamj/whisper-large-carib
# CUDA 12.8 build of PyTorch: the first CUDA line that ships kernels for
# Blackwell GPUs (RTX 5090 / B200). The cu126 build crashed on those with
# "no kernel image is available for execution on the device".
# Needs a host driver >= 570, so the endpoint's allowed CUDA versions must be 12.8+.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    MODEL_ID=neddamj/whisper-large-carib \
    MODEL_DIR=/models/whisper-large-carib

# ffmpeg is used by the transformers ASR pipeline to decode mp3/m4a/ogg/wav/etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the model weights into the image so cold starts don't hit Hugging Face.
# Pin a revision with --build-arg MODEL_REVISION=<commit sha> for reproducible builds.
ARG MODEL_REVISION=9ed32e0fe885b436006af4dafcdf5ead013301a3
COPY download_model.py .
RUN python download_model.py --revision "${MODEL_REVISION}"

COPY handler.py .

# Everything the worker needs is already in the image.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "-u", "handler.py"]
