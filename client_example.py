"""Call the deployed RunPod endpoint.

    set RUNPOD_API_KEY=...        (PowerShell: $env:RUNPOD_API_KEY="...")
    set RUNPOD_ENDPOINT_ID=...
    python client_example.py path/to/audio.mp3
    python client_example.py https://example.com/audio.wav
"""

import base64
import os
import sys

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]


def main(src: str) -> None:
    if src.startswith(("http://", "https://")):
        payload = {"audio": src}
    else:
        with open(src, "rb") as f:
            payload = {"audio_base64": base64.b64encode(f.read()).decode()}
    payload.update({"language": "english", "return_timestamps": True})

    resp = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"input": payload},
        timeout=600,
    )
    resp.raise_for_status()
    print(resp.json())


if __name__ == "__main__":
    main(sys.argv[1])
