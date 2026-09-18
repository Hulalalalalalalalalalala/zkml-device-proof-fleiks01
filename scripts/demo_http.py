"""Show the existing HTTP product API, then stop this demonstration server."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from device_proof.api import app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Demonstrate the local scoring API.")
    parser.add_argument("--port", type=int, default=4311)
    args = parser.parse_args()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="info"))
    thread = threading.Thread(target=server.run, name="device-score-demo")
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("The local demonstration server did not start")
            time.sleep(0.05)
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=10) as client:
            for path in ["/healthz", "/models"]:
                response = client.get(path)
                print(f"GET {path} -> {response.status_code}", flush=True)
                print(json.dumps(response.json(), indent=2, ensure_ascii=False), flush=True)
                response.raise_for_status()
            sample = json.loads((ROOT / "examples" / "sample.json").read_text())
            response = client.post("/score", json=sample)
            print(f"POST /score -> {response.status_code}", flush=True)
            print(json.dumps(response.json(), indent=2, ensure_ascii=False), flush=True)
            response.raise_for_status()
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("The local demonstration server did not stop")


if __name__ == "__main__":
    main()
