"""Integrity audit, fail-closed guard, HTTP endpoint and CLI tests."""

import copy
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from device_proof.api import app
from device_proof.integrity import (
    MANIFEST_CONTRACT,
    MANIFEST_INVALID,
    MODEL_CONTRACT_VIOLATED,
    MODEL_DIGEST_MISMATCH,
    UNKNOWN_MODEL,
    IntegrityError,
    audit_snapshot,
)
from device_proof.scoring import MODEL_ID, ROOT, guard

MODEL_PATH = ROOT / "models" / f"{MODEL_ID}.onnx"
MANIFEST_PATH = ROOT / "models" / f"{MODEL_ID}.json"
SAMPLE = json.loads((ROOT / "examples" / "sample.json").read_text())


@pytest.fixture
def client():
    guard._entry = None
    yield TestClient(app)
    guard._entry = None


@pytest.fixture
def artifacts():
    """Back up the released artifacts and always restore them verbatim."""
    model_backup = MODEL_PATH.read_bytes()
    manifest_backup = MANIFEST_PATH.read_bytes()
    guard._entry = None
    yield MODEL_PATH, MANIFEST_PATH
    MODEL_PATH.write_bytes(model_backup)
    MANIFEST_PATH.write_bytes(manifest_backup)
    guard._entry = None


def valid_manifest_bytes() -> bytes:
    return MANIFEST_PATH.read_bytes()


# --- Audit of the released artifacts ---------------------------------------


def test_released_artifact_passes_audit():
    report = audit_snapshot(
        MODEL_ID, MODEL_PATH.read_bytes(), valid_manifest_bytes()
    )
    assert report["status"] == "verified"
    assert report["model_id"] == MODEL_ID
    assert report["mode"] == "ordinary-inference"
    assert report["sha256"] == sha256(MODEL_PATH.read_bytes()).hexdigest()
    assert report["feature_order"] == [
        "temperature",
        "vibration",
        "current",
        "runtime",
    ]
    assert report["feature_range"] == [0.0, 1.0]
    assert report["score_range"] == [0.1, 1.0]
    assert report["input_shape"] == [1, 4]
    assert report["output_shape"] == [1, 1]
    assert report["onnx_opset"] == 13
    assert report["onnx_ir_version"] == 8
    assert report["inputs"] == [
        {"name": "features", "dtype": "FLOAT", "shape": [1, 4]}
    ]
    assert report["outputs"] == [
        {"name": "score", "dtype": "FLOAT", "shape": [1, 1]}
    ]
    assert report["operators"] == ["MatMul", "Add"]
    assert {init["name"] for init in report["initializers"]} == {"weights", "bias"}
    assert [case["case"] for case in report["golden_inference"]] == [
        "zeros",
        "example",
        "ones",
    ]


def test_unknown_model_audit():
    with pytest.raises(IntegrityError) as exc:
        audit_snapshot("other", MODEL_PATH.read_bytes(), valid_manifest_bytes())
    assert exc.value.code == UNKNOWN_MODEL


# --- Manifest validation ----------------------------------------------------


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda m: m.update({"onnx_opset": "13"}), MANIFEST_INVALID),
        (lambda m: m.update({"input_shape": [1, "4"]}), MANIFEST_INVALID),
        (lambda m: m.update({"feature_range": [0.0]}), MANIFEST_INVALID),
        (lambda m: m.pop("version"), MANIFEST_CONTRACT),
        (lambda m: m.update({"extra": 1}), MANIFEST_CONTRACT),
        (lambda m: m.update({"id": "device-health-v2"}), MANIFEST_CONTRACT),
        (lambda m: m.update({"onnx_opset": 12}), MANIFEST_CONTRACT),
        (lambda m: m.update({"onnx_ir_version": 7}), MANIFEST_CONTRACT),
        (
            lambda m: m.update(
                {"feature_order": ["temperature", "vibration", "current", "current"]}
            ),
            MANIFEST_CONTRACT,
        ),
        (
            lambda m: m.update(
                {"feature_order": ["temperature", "vibration", "current", "evil"]}
            ),
            MANIFEST_CONTRACT,
        ),
        (
            lambda m: m.update(
                {"feature_order": ["runtime", "vibration", "current", "temperature"]}
            ),
            MANIFEST_CONTRACT,
        ),
        (lambda m: m.update({"input_shape": [1, 3]}), MANIFEST_CONTRACT),
        (lambda m: m.update({"output_shape": [1, 2]}), MANIFEST_CONTRACT),
        (lambda m: m.update({"feature_range": [0.0, 0.9]}), MANIFEST_CONTRACT),
        (lambda m: m.update({"score_range": [0.0, 1.0]}), MANIFEST_CONTRACT),
        (lambda m: m.update({"sha256": "z" * 64}), MANIFEST_INVALID),
    ],
)
def test_manifest_validation(mutate, code):
    manifest = json.loads(valid_manifest_bytes())
    mutate(manifest)
    with pytest.raises(IntegrityError) as exc:
        audit_snapshot(
            MODEL_ID, MODEL_PATH.read_bytes(), json.dumps(manifest).encode()
        )
    assert exc.value.code == code


@pytest.mark.parametrize("raw", [b"not json", b"\xff\xfe", b"[]", b"null"])
def test_unparseable_manifest(raw):
    with pytest.raises(IntegrityError) as exc:
        audit_snapshot(MODEL_ID, MODEL_PATH.read_bytes(), raw)
    assert exc.value.code == MANIFEST_INVALID


def test_digest_mismatch():
    manifest = json.loads(valid_manifest_bytes())
    manifest["sha256"] = "0" * 64
    with pytest.raises(IntegrityError) as exc:
        audit_snapshot(
            MODEL_ID, MODEL_PATH.read_bytes(), json.dumps(manifest).encode()
        )
    assert exc.value.code == MODEL_DIGEST_MISMATCH


def test_garbage_model_bytes():
    with pytest.raises(IntegrityError) as exc:
        audit_snapshot(MODEL_ID, b"definitely not onnx", valid_manifest_bytes())
    assert exc.value.code in {"model_digest_mismatch", "model_parse_failed"}


# --- HTTP endpoint ----------------------------------------------------------


def test_integrity_endpoint_success(client):
    response = client.get(f"/models/{MODEL_ID}/integrity")
    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert response.json()["sha256"]


def test_integrity_endpoint_unknown_model(client):
    response = client.get("/models/absent/integrity")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown model"}


def test_integrity_endpoint_unavailable_when_tampered(client, artifacts):
    MODEL_PATH.write_bytes(b"tampered artifact body")
    response = client.get(f"/models/{MODEL_ID}/integrity")
    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}


def test_score_blocked_when_artifact_invalid(client, artifacts):
    # Warm the cache against the valid artifact first.
    warm = client.post("/score", json=SAMPLE)
    assert warm.status_code == 200
    # Replace the file in place; the cached session must not keep serving.
    MODEL_PATH.write_bytes(b"tampered artifact body")
    response = client.post("/score", json=SAMPLE)
    assert response.status_code == 503
    assert response.json() == {"detail": "model_unavailable"}
    # Every other artifact-touching route fails closed as well.
    assert client.get("/healthz").status_code == 503
    assert client.get("/models").status_code == 503
    assert client.get(f"/models/{MODEL_ID}").status_code == 503


def test_service_recovers_after_restore(client, artifacts):
    valid_bytes = MODEL_PATH.read_bytes()
    MODEL_PATH.write_bytes(b"tampered artifact body")
    assert client.post("/score", json=SAMPLE).status_code == 503
    MODEL_PATH.write_bytes(valid_bytes)
    response = client.post("/score", json=SAMPLE)
    assert response.status_code == 200
    assert abs(response.json()["score"] - 0.4) <= 1e-6
    assert client.get(f"/models/{MODEL_ID}/integrity").status_code == 200


def test_unknown_model_routes_remain_404(client):
    assert client.get("/models/absent/integrity").status_code == 404
    payload = copy.deepcopy(SAMPLE)
    payload["model_id"] = "absent"
    assert client.post("/score", json=payload).status_code == 404
    assert client.get("/models/absent").status_code == 404


# --- Guard: replacement sensitivity and concurrency ------------------------


def test_guard_revalidates_replacement_then_restore(artifacts):
    valid_bytes = MODEL_PATH.read_bytes()
    entry = guard.access()
    assert entry.model_digest == sha256(valid_bytes).hexdigest()

    MODEL_PATH.write_bytes(b"replaced while session cached")
    with pytest.raises(IntegrityError):
        guard.access()

    MODEL_PATH.write_bytes(valid_bytes)
    entry = guard.access()
    values = [[0.2, 0.3, 0.4, 0.5]]
    import numpy as np

    out = entry.session.run(["score"], {"features": np.array(values, dtype=np.float32)})[
        0
    ]
    assert abs(float(out[0, 0]) - 0.4) <= 1e-6


def test_concurrent_access_never_exposes_partial_results(artifacts):
    import threading

    import numpy as np

    valid_bytes = MODEL_PATH.read_bytes()
    observed = []
    stop = threading.Event()

    def worker():
        feed = np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
        while not stop.is_set():
            try:
                entry = guard.access()
                value = float(
                    entry.session.run(["score"], {"features": feed})[0][0, 0]
                )
                observed.append(value)
            except IntegrityError as error:
                observed.append(("error", error.code))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for _ in range(200):
        MODEL_PATH.write_bytes(b"mid-check replacement")
        MODEL_PATH.write_bytes(valid_bytes)
    stop.set()
    for thread in threads:
        thread.join()

    assert observed, "workers produced no observations"
    for item in observed:
        if isinstance(item, tuple):
            assert item[0] == "error"
        else:
            # A published result must always be the verified value, never a
            # partial/garbage computation.
            assert abs(item - 0.4) <= 1e-6


# --- CLI --------------------------------------------------------------------


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "device_proof", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_cli_model_check_success():
    result = _run_cli("model-check")
    assert result.returncode == 0
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["status"] == "verified"
    assert report["model_id"] == MODEL_ID


def test_cli_model_check_unknown_model():
    result = _run_cli("model-check", "--model-id", "absent")
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip() == UNKNOWN_MODEL


def test_cli_model_check_failure_leaks_nothing(artifacts):
    MODEL_PATH.write_bytes(b"tampered artifact body")
    result = _run_cli("model-check")
    assert result.returncode != 0
    assert result.stdout == ""
    # stderr is a single stable code only.
    assert result.stderr.strip() == MODEL_DIGEST_MISMATCH
    assert result.stderr.count("\n") == 1
    lowered = result.stderr.lower()
    for leaked in ("traceback", ".onnx", "models/", "file", "error:"):
        assert leaked not in lowered


def test_legacy_cli_still_scores():
    result = _run_cli("--input", "examples/sample.json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "ordinary-inference"
    assert abs(payload["score"] - 0.4) <= 1e-6
