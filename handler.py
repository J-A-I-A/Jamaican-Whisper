"""RunPod Serverless handler for neddamj/whisper-large-carib (Whisper large-v3 fine-tune).

Input (job["input"]):
    audio             str   URL of an audio file (http/https)          } one of
    audio_base64      str   base64-encoded audio file bytes             } these
    language          str   default "english"; null to auto-detect
    task              str   "transcribe" (default) or "translate"
    return_timestamps bool|"word"  default False
    chunk_length_s    int   default 30; 0 disables chunking (sequential long-form)
    batch_size        int   default 8 (chunks decoded in parallel)
    num_beams         int   default 1

Output:
    {"text": "...", "chunks": [...] (only when timestamps requested), "model": "..."}
"""

import base64
import os
import time

import requests
import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.environ.get("MODEL_ID", "neddamj/whisper-large-carib")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/whisper-large-carib")
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", 500 * 1024 * 1024))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def load_pipeline():
    source = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID
    start = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        source,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(source)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=DEVICE,
    )
    print(f"Loaded {source} on {DEVICE} ({DTYPE}) in {time.time() - start:.1f}s")
    return asr


# Loaded once per worker, outside the handler, so warm requests skip model load.
ASR = load_pipeline()


def fetch_audio(job_input: dict) -> bytes:
    if job_input.get("audio_base64"):
        data = job_input["audio_base64"]
        if "," in data[:100] and data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)

    url = job_input.get("audio")
    if not url:
        raise ValueError("Provide either 'audio' (URL) or 'audio_base64'.")
    if not url.startswith(("http://", "https://")):
        raise ValueError("'audio' must be an http(s) URL.")

    with requests.get(url, stream=True, timeout=(10, 120)) as resp:
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=1 << 20):
            buf.extend(chunk)
            if len(buf) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"Audio exceeds {MAX_DOWNLOAD_BYTES} bytes.")
        return bytes(buf)


def handler(job):
    job_input = job.get("input") or {}
    try:
        audio_bytes = fetch_audio(job_input)
    except Exception as e:
        return {"error": f"Could not load audio: {e}"}

    generate_kwargs = {
        "task": job_input.get("task", "transcribe"),
        "num_beams": int(job_input.get("num_beams", 1)),
    }
    language = job_input.get("language", "english")
    if language:
        generate_kwargs["language"] = language

    return_timestamps = job_input.get("return_timestamps", False)
    chunk_length_s = int(job_input.get("chunk_length_s", 30))

    call_kwargs = {
        "return_timestamps": return_timestamps,
        "generate_kwargs": generate_kwargs,
        "batch_size": int(job_input.get("batch_size", 8)),
    }
    if chunk_length_s > 0:
        call_kwargs["chunk_length_s"] = chunk_length_s

    start = time.time()
    try:
        with torch.inference_mode():
            result = ASR(audio_bytes, **call_kwargs)
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    output = {
        "text": result["text"].strip(),
        "model": MODEL_ID,
        "inference_seconds": round(time.time() - start, 3),
    }
    if return_timestamps and "chunks" in result:
        output["chunks"] = [
            {"text": c["text"], "timestamp": list(c["timestamp"])} for c in result["chunks"]
        ]
    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
