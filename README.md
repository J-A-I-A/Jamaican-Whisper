# Jamaican Whisper: RunPod Serverless

A RunPod Serverless worker for [`neddamj/whisper-large-carib`](https://huggingface.co/neddamj/whisper-large-carib), a Whisper large-v3 model fine-tuned for Caribbean English. RunPod builds the image from this repo through its GitHub integration.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Starts from PyTorch 2.8 (CUDA 12.8, so Blackwell GPUs such as the RTX 5090 work), adds ffmpeg, and bakes the model weights into the image |
| `download_model.py` | Downloads the model at build time. It fetches only the latest `model.safetensors` and skips the stale sharded checkpoint and the TensorBoard runs |
| `handler.py` | The RunPod handler. It loads the model once per worker in fp16 and runs the transformers ASR pipeline |
| `test_input.json` | Sample input that RunPod uses when you run `python handler.py` locally |
| `client_example.py` | Example script that calls the deployed endpoint |
| `.github/workflows/ci.yml` | Lightweight lint and syntax checks. The image itself is built by RunPod |

## Deploy

1. **Push this folder to GitHub**
   ```powershell
   git init -b main
   git add .
   git commit -m "RunPod serverless worker for whisper-large-carib"
   git remote add origin https://github.com/<you>/jamaican-whisper-runpod.git
   git push -u origin main
   ```
2. **Connect GitHub to RunPod.** In the RunPod console, go to Settings → Connections → GitHub → Connect, then grant access to the repo.
3. **Create the endpoint.** Go to Serverless → New Endpoint → GitHub Repo, then pick the repo and the `main` branch. Leave Dockerfile path as `Dockerfile` and build context as `.`.
4. **Configure the endpoint**
   - **GPU:** 24 GB or more (for example L4, A5000, RTX 4090, or A10G). The fp16 model uses about 3.5 GB of VRAM, so 16 GB also works.
   - **Container disk:** at least 20 GB. The image is about 13 GB, including the 6.2 GB of fp32 weights.
   - **Execution timeout:** 600 s or more for long audio.
   - **Active workers:** 0 for scale-to-zero, or 1 to avoid cold starts.
   - **FlashBoot:** on.
   - **Allowed CUDA versions:** 12.8 or newer. The image is built on CUDA 12.8, which needs a host driver of 570 or later.
   - **Env vars (optional):** `MAX_DOWNLOAD_BYTES` (default 500 MB).
5. **Redeploy.** Each push to `main` (or a GitHub release, depending on how you configure the endpoint) triggers a rebuild and rollout. For reproducible builds, pin the model revision with the `MODEL_REVISION` build argument, for example `9ed32e0fe885b436006af4dafcdf5ead013301a3`.

## API

`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync` (or `/run` for async and polling `/status/<id>`)

```json
{
  "input": {
    "audio": "https://example.com/clip.mp3",
    "language": "english",
    "task": "transcribe",
    "return_timestamps": false,
    "chunk_length_s": 30,
    "batch_size": 8,
    "num_beams": 1
  }
}
```

- Send either `audio` (an http(s) URL) or `audio_base64`. RunPod caps request payloads at about 10 MB for `/run` and 20 MB for `/runsync`, so use URLs for large files.
- Set `return_timestamps` to `true` for segment timestamps or `"word"` for word-level timestamps.
- Set `language` to `null` to auto-detect.
- Any format ffmpeg can decode works (wav, mp3, m4a, flac, ogg, webm, and so on).

Response:
```json
{ "text": "...", "model": "neddamj/whisper-large-carib", "inference_seconds": 1.23, "chunks": [...] }
```

Example with curl:
```bash
curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"audio": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"}}'
```

## Local test (needs an NVIDIA GPU and Docker)

```bash
docker build -t whisper-carib .
docker run --gpus all --rm whisper-carib python -u handler.py --test_input "$(cat test_input.json)"
```
