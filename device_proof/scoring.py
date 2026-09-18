"""ONNX inference for the bundled demonstration model."""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "device-health-v1.onnx"
MANIFEST_PATH = ROOT / "models" / "device-health-v1.json"
MODEL_ID = "device-health-v1"


class Features(BaseModel):
    """Normalized, synthetic sensor features; larger values mean more stress."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    temperature: float = Field(ge=0, le=1)
    vibration: float = Field(ge=0, le=1)
    current: float = Field(ge=0, le=1)
    runtime: float = Field(ge=0, le=1)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = MODEL_ID
    features: Features


class ScoreResponse(BaseModel):
    model_id: str
    model_sha256: str
    score: float
    mode: str = "ordinary-inference"


def model_info() -> dict:
    info = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    digest = sha256(MODEL_PATH.read_bytes()).hexdigest()
    if info["sha256"] != digest:
        raise RuntimeError("Bundled model digest does not match its manifest")
    return info


@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    model_info()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )


def score(request: ScoreRequest) -> ScoreResponse:
    if request.model_id != MODEL_ID:
        raise KeyError(request.model_id)
    info = model_info()
    feature_order = info["feature_order"]
    values = np.array(
        [[getattr(request.features, feature) for feature in feature_order]],
        dtype=np.float32,
    )
    output = session().run(["score"], {"features": values})[0]
    return ScoreResponse(
        model_id=MODEL_ID,
        model_sha256=info["sha256"],
        score=float(output[0, 0]),
    )
