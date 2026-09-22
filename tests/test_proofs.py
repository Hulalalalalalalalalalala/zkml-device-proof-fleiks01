"""End-to-end tests for local CPU proof jobs and verification.

Every proof exercised here is a genuine EZKL 23.0.5 prove/verify round
trip (see conftest.proof_backend); no mocking or placeholder artifacts are
used.
"""

import base64
import json

import pytest

from device_proof.backend import ProofBackendError
from device_proof import jobs as jobs_mod
from device_proof.jobs import JobManager
from device_proof.proof_materials import (
    b64_decode,
    b64_encode,
    digest_bytes,
    statement_digest_from_manifest,
)

from conftest import BACKEND_DIR, SAMPLE, wait_for


def _wait_http(client, job_id, timeout=40.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/proof-jobs/{job_id}")
        body = resp.json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return resp, body
        time.sleep(0.05)
    raise AssertionError("job never finished")


class _Gate:
    def __init__(self, manager):
        import threading

        self.event = threading.Event()
        self.calls = []
        backend = manager.backend()
        self._backend = backend
        self._real = backend.prove

        def gated(features, job_dir):
            self.calls.append(features)
            self.event.wait(timeout=20)
            return self._real(features, job_dir)

        backend.prove = gated

    def restore(self):
        self._backend.prove = self._real
        self.event.set()


# -- submission / acceptance ------------------------------------------------

def test_submit_returns_202_with_claim_fields(client):
    resp = client.post("/proof-jobs", json=SAMPLE)
    assert resp.status_code == 202
    body = resp.json()
    assert set(body) == {
        "job_id", "status", "queued", "model_id", "model_sha256",
        "quantization_id", "q_score", "scale", "rounding",
        "statement_sha256", "ezkl_version",
    }
    assert body["status"] == "queued"
    assert body["queued"] is True
    assert isinstance(body["job_id"], str) and len(body["job_id"]) == 32
    assert body["model_id"] == "device-health-v1"
    assert body["quantization_id"] == "device-health-v1-q16-v1"
    assert body["ezkl_version"] == "23.0.5"
    assert body["q_score"] == 26214
    assert len(body["statement_sha256"]) == 64


def test_submit_reuses_score_request_validation(client):
    bad = dict(SAMPLE, features={**SAMPLE["features"], "temperature": 2.0})
    assert client.post("/proof-jobs", json=bad).status_code == 422
    missing = {"model_id": "device-health-v1"}
    assert client.post("/proof-jobs", json=missing).status_code == 422
    unknown = dict(SAMPLE, model_id="absent")
    resp = client.post("/proof-jobs", json=unknown)
    assert resp.status_code == 404


# -- full real proof --------------------------------------------------------

def test_real_proof_job_succeeds_with_materials(client):
    submit = client.post("/proof-jobs", json=SAMPLE)
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    resp, body = _wait_http(client, job_id)
    assert resp.status_code == 200
    assert body["status"] == "succeeded"
    assert body["error_code"] is None
    materials = body["materials"]
    assert set(materials) == {
        "manifest", "proof", "verification_key", "settings", "instances",
    }
    manifest = materials["manifest"]
    assert manifest["model_sha256"] == submit.json()["model_sha256"]
    assert manifest["quantization_id"] == "device-health-v1-q16-v1"
    assert manifest["q_score"] == 26214
    assert manifest["statement_sha256"] == submit.json()["statement_sha256"]
    assert manifest["ezkl_version"] == "23.0.5"
    assert manifest["circuit"]["input_visibility"] == "Private"
    assert manifest["circuit"]["output_visibility"] == "Public"
    # Public instances carry exactly one output felt and no private inputs.
    assert len(materials["instances"]) == 1
    assert len(materials["instances"][0]) == 1
    felt = manifest["public_output"]["felt"]
    assert abs(felt - manifest["q_score"]) <= 1
    # Every material digest in the manifest matches the supplied bytes.
    assert manifest["materials"]["proof_sha256"] == \
        digest_bytes(b64_decode(materials["proof"]))
    assert manifest["materials"]["verification_key_sha256"] == \
        digest_bytes(b64_decode(materials["verification_key"]))
    assert manifest["materials"]["settings_sha256"] == \
        digest_bytes(b64_decode(materials["settings"]))


def test_responses_never_leak_sensitive_data(client):
    job_id = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
    _, body = _wait_http(client, job_id)
    text = json.dumps(body)
    for forbidden in (
        "features", "witness", "temperature", "vibration", "current",
        "runtime", "q_t", "q_v", "q_c", "q_r", "input_data",
    ):
        assert forbidden not in text


def test_queued_job_view_has_no_materials_field(client):
    job_id = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
    body = client.get(f"/proof-jobs/{job_id}").json()
    assert body["status"] in ("queued", "running", "succeeded")
    if body["status"] != "succeeded":
        assert "materials" not in body


# -- FIFO, cancellation, status codes --------------------------------------

def test_fifo_order_and_cancel_only_queued(manager, client):
    gate = _Gate(manager)
    try:
        first = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
        # Wait until the single worker picked up the first job.
        import time
        deadline = time.time() + 20
        while client.get(f"/proof-jobs/{first}").json()["status"] != "running":
            time.sleep(0.02)
            assert time.time() < deadline
        second = client.post("/proof-jobs", json={
            "model_id": "device-health-v1",
            "features": {"temperature": 0, "vibration": 0, "current": 0, "runtime": 0},
        }).json()["job_id"]
        assert client.get(f"/proof-jobs/{second}").json()["status"] == "queued"

        # Unknown job -> 404.
        assert client.delete("/proof-jobs/" + "f" * 32).status_code == 404
        # Running -> 409.
        running_del = client.delete(f"/proof-jobs/{first}")
        assert running_del.status_code == 409
        assert running_del.json()["detail"] == "proof_job_not_cancellable"
        # Queued -> 200 cancelled.
        queued_del = client.delete(f"/proof-jobs/{second}")
        assert queued_del.status_code == 200
        assert queued_del.json()["status"] == "cancelled"
        # Terminal states never regress; a second delete is 409.
        assert client.delete(f"/proof-jobs/{second}").status_code == 409

        gate.restore()
        _, first_body = _wait_http(client, first)
        assert first_body["status"] == "succeeded"
        second_body = client.get(f"/proof-jobs/{second}").json()
        assert second_body["status"] == "cancelled"
        # Only the first job ever reached the worker (strict FIFO).
        assert len(gate.calls) == 1
    finally:
        gate.restore()


def test_get_unknown_job_404(client):
    assert client.get("/proof-jobs/" + "a" * 32).status_code == 404


# -- failure code -----------------------------------------------------------

def test_proof_failure_records_stable_code(manager, client):
    def boom(features, job_dir):
        raise RuntimeError("internal rust detail that must not surface")

    manager.backend().prove = boom
    job_id = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
    _, body = _wait_http(client, job_id)
    assert body["status"] == "failed"
    assert body["error_code"] == "proof_generation_failed"
    text = json.dumps(body)
    assert "rust detail" not in text
    # A failed job exposes no materials and cannot be cancelled.
    assert "materials" not in body
    assert client.delete(f"/proof-jobs/{job_id}").status_code == 409


# -- restart -> interrupted, resubmittable ---------------------------------

def test_restart_marks_inflight_interrupted_and_allows_resubmit(tmp_path):
    root = tmp_path / "rt"
    first = JobManager(root, backend_dir=BACKEND_DIR)
    gate = _Gate(first)
    try:
        from device_proof.scoring import ScoreRequest

        r1 = first.submit(ScoreRequest.model_validate(SAMPLE))
        r2 = first.submit(ScoreRequest.model_validate({
            "model_id": "device-health-v1",
            "features": {"temperature": 1, "vibration": 1, "current": 1, "runtime": 1},
        }))
        import time
        deadline = time.time() + 20
        while first.get(r1["job_id"])["status"] != "running":
            time.sleep(0.02)
            assert time.time() < deadline
        # A new process loading the store can only see interrupted jobs: the
        # sensitive inputs of queued/running jobs lived in the old memory.
        second = JobManager(root)
        b1 = second.get(r1["job_id"])
        b2 = second.get(r2["job_id"])
        assert b1["status"] == "failed" and b1["error_code"] == "interrupted"
        assert b2["status"] == "failed" and b2["error_code"] == "interrupted"
    finally:
        gate.restore()

    # Same request is resubmittable on a healthy manager and succeeds.
    from device_proof.scoring import ScoreRequest

    healthy = JobManager(tmp_path / "rt2", backend_dir=BACKEND_DIR)
    rec = healthy.submit(ScoreRequest.model_validate(SAMPLE))
    deadline = __import__("time").time() + 40
    while healthy.get(rec["job_id"])["status"] not in \
            ("succeeded", "failed", "cancelled"):
        __import__("time").sleep(0.05)
        assert __import__("time").time() < deadline
    assert healthy.get(rec["job_id"])["status"] == "succeeded"


# -- backend unavailable -> 503 --------------------------------------------

def test_no_backend_returns_503(monkeypatch, tmp_path, client):
    manager = jobs_mod.get_manager()
    manager._backend = None
    manager._backend_error = None

    class BrokenBackend:
        def __init__(self, *args, **kwargs):
            raise ProofBackendError("proof_backend_unavailable")

    monkeypatch.setattr(jobs_mod, "EzklBackend", BrokenBackend)
    resp = client.post("/proof-jobs", json=SAMPLE)
    assert resp.status_code == 503
    assert resp.json()["detail"] == "proof_backend_unavailable"
    resp = client.post("/proof-verifications", json={
        "manifest": "{}", "proof": "", "verification_key": "", "settings": ""})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "proof_backend_unavailable"


# -- verification -----------------------------------------------------------

def test_verify_valid_material(client, good_payload):
    payload, _materials = good_payload
    resp = client.post("/proof-verifications", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["verified"] is True
    assert body["quantization_id"] == "device-health-v1-q16-v1"
    assert body["q_score"] == 26214
    assert len(body["instances"][0]) == 1
    assert len(body["statement_sha256"]) == 64


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(proof="not-base64!!"),
        lambda p: p.update(proof=""),  # present but undecodable
        lambda p: p.update(verification_key=b64_encode(b"wrong key bytes")),
        lambda p: p.update(settings=b64_encode(b"{}")),
        lambda p: p.update(manifest="{not json"),
        lambda p: p.update(manifest=b64_encode(b"{}")),  # JSON string, not object text
    ],
)
def test_verify_rejects_structurally_invalid_material(client, good_payload, mutate):
    payload, _ = good_payload
    payload = json.loads(json.dumps(payload))
    mutate(payload)
    resp = client.post("/proof-verifications", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_proof_material"


def test_verify_rejects_missing_and_extra_fields(client, good_payload):
    payload, _ = good_payload
    missing = {k: v for k, v in payload.items() if k != "proof"}
    assert client.post("/proof-verifications", json=missing).status_code == 422
    extra = dict(payload, bogus=1)
    assert client.post("/proof-verifications", json=extra).status_code == 422


def test_verify_rejects_digest_tampering(client, good_payload):
    payload, _ = good_payload
    payload = json.loads(json.dumps(payload))
    manifest = json.loads(payload["manifest"])
    manifest["q_score"] = 999
    payload["manifest"] = json.dumps(manifest)
    resp = client.post("/proof-verifications", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_proof_material"


def test_verify_rejects_cryptographically_forged_proof(client, good_payload):
    # Re-sign nothing: mutate proof bytes, then repair only the manifest's
    # proof digest so offline digest checks pass and EZKL itself must reject.
    payload, _ = good_payload
    payload = json.loads(json.dumps(payload))
    raw = bytearray(b64_decode(payload["proof"]))
    proof_doc = json.loads(bytes(raw))
    proof_doc["proof"][0] ^= 0x01
    forged = json.dumps(proof_doc, separators=(",", ":")).encode("utf-8")
    payload["proof"] = b64_encode(forged)
    manifest = json.loads(payload["manifest"])
    manifest["materials"]["proof_sha256"] = digest_bytes(forged)
    payload["manifest"] = json.dumps(manifest)
    resp = client.post("/proof-verifications", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_proof_material"


def test_verify_rejects_wrong_model_mapping(client, good_payload):
    # Internally consistent statement digests, but bound to a different model.
    payload, _ = good_payload
    payload = json.loads(json.dumps(payload))
    manifest = json.loads(payload["manifest"])
    manifest["model_sha256"] = "0" * 64
    manifest["statement_sha256"] = statement_digest_from_manifest(manifest)
    payload["manifest"] = json.dumps(manifest)
    resp = client.post("/proof-verifications", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_proof_material"


def test_verify_rejects_relabeled_quantization(client, good_payload):
    # A genuine proof relabeled as a different quantization scheme (with a
    # self-consistent recomputed statement digest) must still be rejected.
    payload, _ = good_payload
    payload = json.loads(json.dumps(payload))
    manifest = json.loads(payload["manifest"])
    manifest["quantization_id"] = "other-quantization-v9"
    manifest["statement_sha256"] = statement_digest_from_manifest(manifest)
    payload["manifest"] = json.dumps(manifest)
    resp = client.post("/proof-verifications", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_proof_material"


def test_distinct_inputs_produce_distinct_verified_claims(client, manager):
    other = {
        "model_id": "device-health-v1",
        "features": {"temperature": 0, "vibration": 0, "current": 0, "runtime": 0},
    }
    job_id = client.post("/proof-jobs", json=other).json()["job_id"]
    _, body = _wait_http(client, job_id)
    assert body["status"] == "succeeded"
    assert body["materials"]["manifest"]["q_score"] == 6554
    payload = {
        "manifest": json.dumps(body["materials"]["manifest"]),
        "proof": body["materials"]["proof"],
        "verification_key": body["materials"]["verification_key"],
        "settings": body["materials"]["settings"],
    }
    verified = client.post("/proof-verifications", json=payload).json()
    assert verified["verified"] is True
    assert verified["q_score"] == 6554


def test_materials_survive_restart_and_remain_verifiable(tmp_path):
    root = tmp_path / "rt"
    first = JobManager(root, backend_dir=BACKEND_DIR)
    from device_proof.scoring import ScoreRequest

    rec = first.submit(ScoreRequest.model_validate(SAMPLE))
    done = wait_for(first, rec["job_id"])
    assert done["status"] == "succeeded"
    # A fresh process reopens the same store; succeeded jobs keep materials.
    second = JobManager(root, backend_dir=BACKEND_DIR)
    reopened = second.get(rec["job_id"])
    assert reopened["status"] == "succeeded"
    assert reopened["materials"]["manifest"]["q_score"] == 26214
    payload = {
        "manifest": json.dumps(reopened["materials"]["manifest"]),
        "proof": reopened["materials"]["proof"],
        "verification_key": reopened["materials"]["verification_key"],
        "settings": reopened["materials"]["settings"],
    }
    assert second.verify_payload(payload)["verified"] is True


def test_no_sensitive_data_persisted_to_disk(tmp_path):
    root = tmp_path / "rt"
    mgr = JobManager(root, backend_dir=BACKEND_DIR)
    from device_proof.scoring import ScoreRequest

    rec = mgr.submit(ScoreRequest.model_validate(SAMPLE))
    done = wait_for(mgr, rec["job_id"])
    assert done["status"] == "succeeded"
    job_id = rec["job_id"]
    persisted = b""
    for path in root.rglob("*"):
        if path.is_file():
            persisted += path.read_bytes()
    text = persisted.decode("utf-8", errors="ignore")
    # No request shape, no witness, and none of the distinctive per-feature
    # Q16 encodings (0.2/0.3/0.5 inputs) nor their field-element hex felts.
    for token in (
        "input_data", "witness", "features",
        "13107", "19661", "32768",
        "6606000000000000000000000000000000000000000000000000000000000000",
        "9a09000000000000000000000000000000000000000000000000000000000000",
        "0010000000000000000000000000000000000000000000000000000000000000",
    ):
        assert token not in text, token
    # No per-job witness/input files remain after the terminal state.
    assert not list((root / "materials" / job_id).rglob("witness*"))
    assert not list((root / "materials" / job_id).rglob("input*"))


def test_succeeded_job_has_no_private_instances(client):
    job_id = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
    _, body = _wait_http(client, job_id)
    proof = json.loads(b64_decode(body["materials"]["proof"]))
    # input_visibility Private means instances hold outputs only, no inputs.
    assert proof["pretty_public_inputs"]["rescaled_inputs"] == []
    assert proof["pretty_public_inputs"]["inputs"] == []
    assert len(proof["pretty_public_inputs"]["outputs"][0]) == 1
