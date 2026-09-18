import copy
import json

from fastapi.testclient import TestClient
import pytest

from device_proof.api import app
from device_proof.scoring import ROOT


client = TestClient(app)
SAMPLE = json.loads((ROOT / "examples" / "sample.json").read_text())


def test_live_model_and_sample_score():
    assert client.get("/healthz").json()["status"] == "ok"
    info = client.get("/models/device-health-v1").json()
    result = client.post("/score", json=SAMPLE)
    assert result.status_code == 200
    assert result.json()["score"] == pytest.approx(0.4, abs=1e-6)
    assert result.json()["model_sha256"] == info["sha256"]
    assert result.json()["mode"] == "ordinary-inference"


@pytest.mark.parametrize("value", [-0.1, 1.01])
def test_out_of_range_features(value):
    payload = copy.deepcopy(SAMPLE)
    payload["features"]["temperature"] = value
    assert client.post("/score", json=payload).status_code == 422


def test_missing_feature_and_unknown_model():
    payload = copy.deepcopy(SAMPLE)
    del payload["features"]["runtime"]
    assert client.post("/score", json=payload).status_code == 422
    payload = copy.deepcopy(SAMPLE)
    payload["model_id"] = "absent"
    assert client.post("/score", json=payload).status_code == 404
    assert client.get("/models/absent").status_code == 404


@pytest.mark.parametrize("value,expected", [(0, 0.1), (1, 1.0)])
def test_feature_boundaries(value, expected):
    payload = copy.deepcopy(SAMPLE)
    payload["features"] = dict.fromkeys(payload["features"], value)
    assert client.post("/score", json=payload).json()["score"] == pytest.approx(expected, abs=1e-6)
