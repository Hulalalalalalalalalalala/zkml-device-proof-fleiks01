"""ONNX inference for the bundled demonstration model."""

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, ConfigDict, Field

from .audit import MODEL_ID, ROOT, audited_artifacts


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
    """Return the audited release manifest for the bundled model."""
    manifest, _session = audited_artifacts(MODEL_ID)
    return manifest


def session() -> ort.InferenceSession:
    """Return the audited inference session, creating it if needed."""
    _manifest, session = audited_artifacts(MODEL_ID)
    return session


def score(request: ScoreRequest) -> ScoreResponse:
    if request.model_id != MODEL_ID:
        raise KeyError(request.model_id)
    manifest, active_session = audited_artifacts(MODEL_ID)
    feature_order = manifest["feature_order"]
    values = np.array(
        [[getattr(request.features, feature) for feature in feature_order]],
        dtype=np.float32,
    )
    output = active_session.run(["score"], {"features": values})[0]
    return ScoreResponse(
        model_id=MODEL_ID,
        model_sha256=manifest["sha256"],
        score=float(output[0, 0]),
    )
