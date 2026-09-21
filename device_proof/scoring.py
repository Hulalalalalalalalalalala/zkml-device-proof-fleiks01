"""ONNX ordinary inference for the fixed demonstration model.

All access to the model artifacts is gated by the integrity audit in
:mod:`device_proof.integrity`, so scoring can only run against an artifact that
matches the published release contract. Nothing here generates or verifies
zero-knowledge proofs: the engine is ordinary CPU inference.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .integrity import (
    FEATURE_ORDER,
    INPUT_NAME,
    MODEL_ID,
    OUTPUT_NAME,
    ModelGuard,
)


ROOT = Path(__file__).resolve().parent.parent

# A single process-wide guard gates the bundled model.
guard = ModelGuard(ROOT)


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
    """Return the audited manifest. Fails closed if the artifact is invalid."""
    entry = guard.access()
    return dict(entry.manifest)


def session():
    """Return a live inference session for an audited artifact."""
    return guard.access().session


def score(request: ScoreRequest) -> ScoreResponse:
    if request.model_id != MODEL_ID:
        raise KeyError(request.model_id)
    entry = guard.access()
    values = np.array(
        [[getattr(request.features, feature) for feature in FEATURE_ORDER]],
        dtype=np.float32,
    )
    output = entry.session.run([OUTPUT_NAME], {INPUT_NAME: values})[0]
    return ScoreResponse(
        model_id=MODEL_ID,
        model_sha256=entry.model_digest,
        score=float(output[0, 0]),
    )
