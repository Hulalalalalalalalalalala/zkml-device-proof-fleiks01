import copy
import json
import time

from fastapi.testclient import TestClient
import pytest

from device_proof import proofs
from device_proof.api import app
from device_proof.canonical import canonical_sha256
from device_proof.scoring import ROOT
from device_proof.statements import QUANTIZATION_ID, SCALE


client = TestClient(app)
SAMPLE = json.loads((ROOT / "examples" / "sample.json").read_text())

STATEMENT_KEYS = {
    "model_id",
    "model_sha256",
    "quantization_id",
    "q_score",
    "scale",
    "rounding",
    "statement_sha256",
}
TERMINAL = {"succeeded", "failed", "cancelled"}


def _submit(payload=None):
    response = client.post("/proof-jobs", json=payload or SAMPLE)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert set(body) == {"job_id", "status"} | STATEMENT_KEYS
    return body


def _wait_terminal(job_id, timeout=180.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/proof-jobs/{job_id}").json()
        if body["status"] in TERMINAL:
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


@pytest.fixture(scope="module")
def succeeded_job():
    body = _submit()
    final = _wait_terminal(body["job_id"])
    assert final["status"] == "succeeded"
    assert final["error_code"] is None
    return final


def test_submit_returns_202_with_statement_fields():
    body = _submit()
    info = client.get("/models/device-health-v1").json()
    assert body["model_id"] == "device-health-v1"
    assert body["model_sha256"] == info["sha256"]
    assert body["quantization_id"] == QUANTIZATION_ID
    assert body["q_score"] == 26214
    assert body["scale"] == SCALE
    assert body["rounding"] == "ties-to-even"
    assert len(body["statement_sha256"]) == 64
    _wait_terminal(body["job_id"])


def test_job_succeeds_with_bound_material(succeeded_job):
    materials = succeeded_job["materials"]
    assert set(materials) == {"manifest", "proof", "settings", "instances", "vk"}
    manifest = materials["manifest"]
    assert manifest["model_sha256"] == succeeded_job["model_sha256"]
    assert manifest["quantization_id"] == QUANTIZATION_ID
    assert manifest["q_score"] == succeeded_job["q_score"]
    assert manifest["statement_sha256"] == succeeded_job["statement_sha256"]
    assert manifest["ezkl_version"] == proofs.EZKL_VERSION
    digests = manifest["materials"]
    assert set(digests) == {"proof", "settings", "vk", "instances"}
    assert canonical_sha256(materials["proof"]) == digests["proof"]
    assert canonical_sha256(materials["settings"]) == digests["settings"]
    assert canonical_sha256(materials["instances"]) == digests["instances"]
    # Public instances carry exactly one public output, never private inputs.
    instances = materials["instances"]
    assert len(instances) == 1 and len(instances[0]) == 1
    assert proofs.ezkl.felt_to_int(instances[0][0]) == pytest.approx(
        succeeded_job["q_score"], abs=proofs.INSTANCE_TOLERANCE
    )


def test_terminal_state_never_regresses(succeeded_job):
    final = _wait_terminal(succeeded_job["job_id"])
    assert final["status"] == "succeeded"


def test_verification_accepts_valid_material(succeeded_job):
    response = client.post("/proof-verifications", json=succeeded_job["materials"])
    assert response.status_code == 200
    assert response.json() == {"verified": True}


def _tampered(materials, mutate):
    payload = copy.deepcopy(materials)
    mutate(payload)
    return client.post("/proof-verifications", json=payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["manifest"].update(q_score=p["manifest"]["q_score"] + 1),
        lambda p: p["manifest"]["materials"].update(proof="0" * 64),
        lambda p: p["proof"].update(proof=p["proof"]["proof"][:-8]),
        lambda p: p["proof"].__setitem__("unexpected", 1),
        lambda p: p.update(unexpected=1),
        lambda p: p["instances"].append(["00" * 32]),  # private inputs must not appear
        lambda p: p.update(vk="not-base64!!"),
        lambda p: p["manifest"].update(ezkl_version="0.0.0"),
        lambda p: p["manifest"].update(model_sha256="0" * 64),
    ],
)
def test_verification_rejects_invalid_material(succeeded_job, mutate):
    response = _tampered(succeeded_job["materials"], mutate)
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_proof_material"


def test_verification_rejects_cryptographically_broken_proof(succeeded_job):
    # Digests recomputed so structural checks pass; EZKL must reject it.
    payload = copy.deepcopy(succeeded_job["materials"])
    payload["proof"]["proof"][0] ^= 1
    payload["manifest"]["materials"]["proof"] = canonical_sha256(payload["proof"])
    response = client.post("/proof-verifications", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_proof_material"


def test_unknown_job_get_and_delete_are_404():
    assert client.get("/proof-jobs/" + "0" * 32).status_code == 404
    assert client.delete("/proof-jobs/" + "0" * 32).status_code == 404


def test_cancel_only_queued_jobs():
    proofs.pause_worker()
    try:
        first = _submit()
        second = _submit()
        cancelled = client.delete(f"/proof-jobs/{second['job_id']}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        # Terminal states cannot be cancelled again.
        assert client.delete(f"/proof-jobs/{second['job_id']}").status_code == 409
    finally:
        proofs.resume_worker()
    final = _wait_terminal(first["job_id"])
    assert final["status"] == "succeeded"
    # The cancelled job never leaves its terminal state.
    assert client.get(f"/proof-jobs/{second['job_id']}").json()["status"] == "cancelled"


def test_delete_running_job_conflicts():
    body = _submit()
    deadline = time.monotonic() + 60
    status = ""
    while time.monotonic() < deadline and status != "running":
        status = client.get(f"/proof-jobs/{body['job_id']}").json()["status"]
        time.sleep(0.05)
    assert status == "running"
    assert client.delete(f"/proof-jobs/{body['job_id']}").status_code == 409
    assert _wait_terminal(body["job_id"])["status"] == "succeeded"


def test_restart_marks_unfinished_jobs_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(proofs, "JOBS_DIR", tmp_path)
    job_id = "ab" * 16
    directory = tmp_path / job_id
    directory.mkdir()
    record = {
        "job_id": job_id,
        "status": "running",
        "statement": {"model_id": "device-health-v1", "q_score": 26214},
        "error_code": None,
        "created_at": 1,
        "updated_at": 1,
    }
    (directory / "job.json").write_text(json.dumps(record), encoding="utf-8")
    try:
        proofs.recover_jobs()
        job = proofs.get_job(job_id)
        assert job["status"] == "failed"
        assert job["error_code"] == "interrupted"
        assert job["materials"] is None
        persisted = json.loads((directory / "job.json").read_text())
        assert persisted["status"] == "failed"
        assert persisted["error_code"] == "interrupted"
    finally:
        with proofs._jobs_lock:
            proofs._jobs.pop(job_id, None)


def test_failed_job_reports_stable_code(monkeypatch):
    proofs.pause_worker()
    try:
        body = _submit()
        monkeypatch.setattr(
            proofs, "_execute", lambda job, features: (_ for _ in ()).throw(RuntimeError("x"))
        )
    finally:
        proofs.resume_worker()
    final = _wait_terminal(body["job_id"])
    assert final["status"] == "failed"
    assert final["error_code"] == "proof_generation_failed"
    assert final["materials"] is None


def test_backend_unavailable_returns_503(monkeypatch):
    monkeypatch.setattr(proofs, "ezkl", None)
    response = client.post("/proof-jobs", json=SAMPLE)
    assert response.status_code == 503
    assert response.json()["detail"] == "proof_backend_unavailable"
    response = client.post("/proof-verifications", json={})
    assert response.status_code == 503
    assert response.json()["detail"] == "proof_backend_unavailable"


def test_unknown_model_and_invalid_features():
    payload = copy.deepcopy(SAMPLE)
    payload["model_id"] = "absent"
    assert client.post("/proof-jobs", json=payload).status_code == 404
    payload = copy.deepcopy(SAMPLE)
    payload["features"]["temperature"] = 1.5
    assert client.post("/proof-jobs", json=payload).status_code == 422


def test_responses_never_carry_sensitive_data(succeeded_job):
    for body in (
        client.get(f"/proof-jobs/{succeeded_job['job_id']}").json(),
        client.post("/proof-verifications", json=succeeded_job["materials"]).json(),
    ):
        text = json.dumps(body)
        for forbidden in (
            "features",
            "temperature",
            "vibration",
            "current",
            "runtime",
            "witness",
            "input_data",
            "q_t",
            "q_v",
            "q_c",
            "q_r",
        ):
            assert forbidden not in text


def test_single_fifo_worker():
    assert proofs._worker_thread is not None
    assert proofs._worker_thread.is_alive()
