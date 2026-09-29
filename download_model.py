"""Build-time download of the model weights into the Docker image.

The HF repo contains both a single-file `model.safetensors` (latest checkpoint)
and an older sharded checkpoint (`model-0000X-of-00002.safetensors` + index).
Only the single file is downloaded: transformers loads it in preference to the
shards, and skipping the stale shards saves ~6 GB of image size.
"""

import argparse
import os

from huggingface_hub import snapshot_download

ALLOW_PATTERNS = [
    "*.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors",
]

IGNORE_PATTERNS = [
    "model-*-of-*.safetensors",
    "model.safetensors.index.json",
    "runs/*",
    "training_args.bin",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "neddamj/whisper-large-carib"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default=os.environ.get("MODEL_DIR", "/models/whisper-large-carib"))
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {args.model_id}@{args.revision} to {path}")
    for name in sorted(os.listdir(path)):
        print("  ", name)


if __name__ == "__main__":
    main()
