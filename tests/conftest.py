"""Shared fixtures for the real-EZKL proof tests.

The circuit is built once per machine (cached in the temp directory and
reused across test runs); every proof in the suite is a genuine EZKL
``prove``/``verify`` round trip, never a mock.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from device_proof.api import app
from device_proof.backend import EzklBackend, ProofBackendError
from device_proof.jobs import reset_manager
from device_proof.scoring import ScoreRequest

BACKEND_DIR = Path(tempfile.gettempdir()) / "dp-test-backend"
SAMPLE = {
    "model_id": "device-health-v1",
    "features": {
        "temperature": 0.2,
        "vibration": 0.3,
        "current": 0.4,
        "runtime": 0.5,
    },
}


def wait_for(mgr, job_id, timeout=40.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = mgr.get(job_id)
        if record["status"] in ("succeeded", "failed", "cancelled"):
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {mgr.get(job_id)}")


@pytest.fixture(scope="session")
def proof_backend():
    try:
        backend = EzklBackend(BACKEND_DIR)
        backend.trusted()
    except ProofBackendError:
        pytest.skip("EZKL proof backend unavailable")
    return backend


@pytest.fixture()
def manager(proof_backend, tmp_path):
    # Isolated job store per test; the (expensive) circuit artifacts are
    # shared read-only via BACKEND_DIR.
    yield reset_manager(tmp_path / "jobs", backend_dir=BACKEND_DIR)


@pytest.fixture()
def client(manager):
    # The app resolves the process-wide manager per request, which the
    # ``manager`` fixture has just reset to an isolated job store.
    return TestClient(app)


@pytest.fixture(scope="session")
def good_payload(proof_backend):
    """One real successful proof, packaged exactly like GET materials."""
    jobs_dir = Path(tempfile.mkdtemp(prefix="dp-good-")) / "jobs"
    mgr = reset_manager(jobs_dir, backend_dir=BACKEND_DIR)
    record = mgr.submit(ScoreRequest.model_validate(SAMPLE))
    done = wait_for(mgr, record["job_id"])
    assert done["status"] == "succeeded", done.get("error_code")
    materials = done["materials"]
    payload = {
        "manifest": json.dumps(materials["manifest"]),
        "proof": materials["proof"],
        "verification_key": materials["verification_key"],
        "settings": materials["settings"],
    }
    return payload, materials
