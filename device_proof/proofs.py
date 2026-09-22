"""Asynchronous local CPU proof jobs backed by EZKL.

A proof job takes a validated ``ScoreRequest`` (the same contract as
``POST /score`` and ``POST /statements``), derives its Q16 quantization
statement, and asks EZKL to prove the audited model's execution for those
features on the local CPU. The circuit is calibrated with input/param scale
16 and rebased outputs, so the single public instance is the score at scale
2**16 and must agree with the statement's ``q_score`` within a small
fixed-point tolerance.

Privacy and persistence rules enforced here:

* Raw features, Q16 encodings, and the EZKL witness live only in process
  memory (witness/input bytes are handed to EZKL through anonymous memfd
  files, never the filesystem). They are released when the job reaches a
  terminal state and never appear in responses, logs, or exceptions.
* Only non-sensitive metadata (job id, status, statement fields, timestamps),
  stable error codes, and the public proof material (manifest, proof,
  verification key, settings, public instances) are persisted, under
  ``proof_data/``.
* Job statuses are queued/running/succeeded/failed/cancelled; terminal
  states never regress. On restart, jobs left queued/running are marked
  failed with code ``interrupted`` and may be resubmitted by clients.
"""

import base64
import binascii
import contextlib
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid

from .audit import MODEL_ID, MODEL_PATH, ROOT, AuditError, audited_artifacts
from .canonical import canonical_bytes, canonical_sha256
from .scoring import ScoreRequest
from .statements import (
    QUANTIZATION_ID,
    ROUNDING,
    SCALE,
    UINT32_MAX,
    build_statement,
)

try:
    import ezkl

    EZKL_VERSION = ezkl.__version__
except Exception:  # pragma: no cover - depends on environment
    ezkl = None
    EZKL_VERSION = None

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = (SUCCEEDED, FAILED, CANCELLED)

ERR_PROOF_GENERATION_FAILED = "proof_generation_failed"
ERR_INTERRUPTED = "interrupted"
ERR_BACKEND_UNAVAILABLE = "proof_backend_unavailable"
ERR_INVALID_MATERIAL = "invalid_proof_material"

PROOF_DATA_DIR = ROOT / "proof_data"
JOBS_DIR = PROOF_DATA_DIR / "jobs"
BACKEND_DIR = PROOF_DATA_DIR / "backend"

# The circuit is calibrated at input/param scale 16 with rebased outputs, so
# the public instance is the score at the statement's own scale (2**16). The
# observed fixed-point deviation from the exact rational q_score is at most 1;
# 8 leaves ample room while still rejecting any materially wrong output.
INSTANCE_TOLERANCE = 8

# Public, non-sensitive calibration constants spanning the feature domain.
CALIBRATION_INPUTS = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.2, 0.3, 0.4, 0.5],
    [0.9, 0.1, 0.5, 0.3],
    [0.5, 0.5, 0.5, 0.5],
    [0.1, 0.9, 0.2, 0.8],
    [0.25, 0.75, 0.5, 0.25],
    [0.75, 0.25, 0.5, 0.75],
]

BACKEND_FILES = ("settings.json", "model.compiled", "srs", "vk", "pk")

MANIFEST_KEYS = {
    "model_id",
    "model_sha256",
    "quantization_id",
    "q_score",
    "scale",
    "rounding",
    "statement_sha256",
    "ezkl_version",
    "materials",
}
MATERIAL_DIGEST_KEYS = {"proof", "settings", "vk", "instances"}
HEX64 = frozenset("0123456789abcdef")


class BackendUnavailable(Exception):
    """The EZKL backend cannot serve proof generation or verification."""

    code = ERR_BACKEND_UNAVAILABLE


class MaterialError(Exception):
    """Submitted proof material failed structural, digest, or mapping checks."""

    code = ERR_INVALID_MATERIAL


class _ProofBackendError(Exception):
    """Internal: backend artifacts could not be prepared."""


def backend_available() -> bool:
    return ezkl is not None


@contextlib.contextmanager
def _memfd(data: bytes):
    """Yield a /proc/self/fd path for bytes held in an anonymous in-memory file.

    Used for feature inputs and witnesses so they never touch the filesystem.
    """
    fd = os.memfd_create("device-proof", flags=0)
    try:
        os.write(fd, data)
        os.lseek(fd, 0, os.SEEK_SET)
        yield f"/proc/self/fd/{fd}"
    finally:
        os.close(fd)


def _is_hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _is_job_id(value) -> bool:
    return isinstance(value, str) and len(value) == 32 and set(value) <= HEX64


def _sha256_bytes(data: bytes) -> str:
    from hashlib import sha256

    return sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Backend artifact preparation (non-sensitive, persisted, reused across runs)
# ---------------------------------------------------------------------------

_backend_lock = threading.Lock()


def ensure_backend(model_sha256: str) -> Path:
    """Prepare (once) and return the EZKL backend directory for the model.

    The directory holds settings.json, the compiled circuit, the SRS, and the
    verification/proving keys. These are derived only from the audited public
    model and public calibration constants, and are safe to persist and reuse.
    """
    if ezkl is None:
        raise BackendUnavailable()
    target = BACKEND_DIR / model_sha256
    with _backend_lock:
        if all((target / name).is_file() for name in BACKEND_FILES):
            return target
        # The audit gates backend preparation like any other model use.
        manifest, _session = audited_artifacts(MODEL_ID)
        if manifest["sha256"] != model_sha256:
            raise _ProofBackendError("model digest mismatch")
        BACKEND_DIR.mkdir(parents=True, exist_ok=True)
        staging = BACKEND_DIR / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            run_args = ezkl.PyRunArgs()
            run_args.input_visibility = "private"
            run_args.output_visibility = "public"
            run_args.param_visibility = "fixed"
            raw_settings = staging / "settings.raw.json"
            ezkl.gen_settings(str(MODEL_PATH), str(raw_settings), py_run_args=run_args)
            calibration = staging / "calibration.json"
            calibration.write_text(
                json.dumps({"input_data": CALIBRATION_INPUTS}), encoding="utf-8"
            )
            ezkl.calibrate_settings(
                str(calibration),
                str(MODEL_PATH),
                str(raw_settings),
                target="resources",
                scales=[16],
            )
            settings = json.loads(raw_settings.read_text(encoding="utf-8"))
            (staging / "settings.json").write_bytes(canonical_bytes(settings))
            ezkl.compile_circuit(
                str(MODEL_PATH), str(staging / "model.compiled"), str(staging / "settings.json")
            )
            ezkl.gen_srs(str(staging / "srs"), settings["run_args"]["logrows"])
            ezkl.setup(
                str(staging / "model.compiled"),
                str(staging / "vk"),
                str(staging / "pk"),
                str(staging / "srs"),
            )
            raw_settings.unlink()
            calibration.unlink()
            try:
                os.replace(staging, target)
            except OSError:
                # Another process completed the same backend first; use it.
                shutil.rmtree(staging, ignore_errors=True)
                if not all((target / name).is_file() for name in BACKEND_FILES):
                    raise
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if isinstance(exc, (AuditError, BackendUnavailable)):
                raise
            raise _ProofBackendError("backend preparation failed") from None
        return target


# ---------------------------------------------------------------------------
# Job store: in-memory records persisted as non-sensitive metadata
# ---------------------------------------------------------------------------


class _Job:
    __slots__ = ("job_id", "status", "statement", "error_code", "created_at", "updated_at")

    def __init__(self, job_id, status, statement, error_code, created_at, updated_at):
        self.job_id = job_id
        self.status = status
        self.statement = statement
        self.error_code = error_code
        self.created_at = created_at
        self.updated_at = updated_at


_jobs: dict[str, _Job] = {}
_jobs_lock = threading.Lock()
# Feature values for not-yet-terminal jobs, keyed by job id. Memory only.
_pending_features: dict[str, list[float]] = {}
_work_queue: "queue.Queue[str]" = queue.Queue()
_worker_gate = threading.Event()
_worker_gate.set()
_worker_thread: threading.Thread | None = None
# Serializes every EZKL call (prove/verify/setup) across worker and requests.
_ezkl_lock = threading.RLock()


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _persist_job(job: _Job) -> None:
    directory = _job_dir(job.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "statement": job.statement,
        "error_code": job.error_code,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    staging = directory / "job.json.tmp"
    staging.write_bytes(canonical_bytes(payload))
    os.replace(staging, directory / "job.json")


def _transition(job: _Job, status: str, error_code) -> None:
    """Move a job to a new state; terminal states never regress."""
    with _jobs_lock:
        if job.status in TERMINAL_STATES:
            return
        job.status = status
        job.error_code = error_code
        job.updated_at = int(time.time())
        _persist_job(job)


def _release_features(job_id: str) -> None:
    with _jobs_lock:
        _pending_features.pop(job_id, None)


def submit(request: ScoreRequest) -> dict:
    """Create a queued proof job for a validated score request."""
    # The statement recomputes the fixed-model audit and the Q16 encodings;
    # encodings stay inside build_statement and are never returned here.
    statement = build_statement(request).model_dump()
    if not backend_available():
        raise BackendUnavailable()
    manifest, _session = audited_artifacts(MODEL_ID)
    features = [getattr(request.features, name) for name in manifest["feature_order"]]
    now = int(time.time())
    job = _Job(uuid.uuid4().hex, QUEUED, statement, None, now, now)
    with _jobs_lock:
        _jobs[job.job_id] = job
        _pending_features[job.job_id] = features
        _persist_job(job)
    start_worker()
    _work_queue.put(job.job_id)
    return {"job_id": job.job_id, "status": job.status, **statement}


def _snapshot(job: _Job) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status,
        **job.statement,
        "error_code": job.error_code,
        "materials": None,
    }


def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        result = _snapshot(job)
        succeeded = job.status == SUCCEEDED
    if succeeded:
        result["materials"] = _load_materials(job_id)
    return result


def _load_materials(job_id: str) -> dict:
    directory = _job_dir(job_id)
    return {
        "manifest": json.loads((directory / "manifest.json").read_bytes()),
        "proof": json.loads((directory / "proof.json").read_bytes()),
        "settings": json.loads((directory / "settings.json").read_bytes()),
        "instances": json.loads((directory / "instances.json").read_bytes()),
        "vk": base64.b64encode((directory / "vk.bin").read_bytes()).decode("ascii"),
    }


def cancel_job(job_id: str):
    """Cancel a queued job. Returns the snapshot, False if not queued, None if unknown."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if job.status != QUEUED:
            return False
        job.status = CANCELLED
        job.updated_at = int(time.time())
        _persist_job(job)
        _pending_features.pop(job_id, None)
        return _snapshot(job)


def recover_jobs() -> None:
    """Load persisted jobs; queued/running survivors of a restart become failed/interrupted."""
    if not JOBS_DIR.is_dir():
        return
    for directory in sorted(JOBS_DIR.iterdir()):
        if not directory.is_dir() or not _is_job_id(directory.name):
            continue
        try:
            data = json.loads((directory / "job.json").read_bytes())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("job_id") != directory.name:
            continue
        statement = data.get("statement")
        status = data.get("status")
        if not isinstance(statement, dict) or status not in (
            QUEUED,
            RUNNING,
            SUCCEEDED,
            FAILED,
            CANCELLED,
        ):
            continue
        job = _Job(
            directory.name,
            status,
            statement,
            data.get("error_code"),
            data.get("created_at", 0),
            data.get("updated_at", 0),
        )
        if job.status in (QUEUED, RUNNING):
            # The features needed to run were memory-only and are gone.
            job.status = FAILED
            job.error_code = ERR_INTERRUPTED
            job.updated_at = int(time.time())
            _persist_job(job)
        with _jobs_lock:
            _jobs[job.job_id] = job


# ---------------------------------------------------------------------------
# Single FIFO worker
# ---------------------------------------------------------------------------


def start_worker() -> None:
    global _worker_thread
    with _jobs_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(
                target=_worker_loop, name="proof-worker", daemon=True
            )
            _worker_thread.start()


def pause_worker() -> None:
    """Hold dequeued jobs in the queued state (used by tests and maintenance)."""
    _worker_gate.clear()


def resume_worker() -> None:
    _worker_gate.set()


def _worker_loop() -> None:
    while True:
        job_id = _work_queue.get()
        try:
            _worker_gate.wait()
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None or job.status != QUEUED:
                    job = None
                else:
                    job.status = RUNNING
                    job.updated_at = int(time.time())
                    _persist_job(job)
                    features = _pending_features.get(job_id)
            if job is None:
                continue
            try:
                if features is None:
                    raise _ProofBackendError("job features unavailable")
                with _ezkl_lock:
                    _execute(job, features)
            except Exception:
                # Only the stable code is recorded; exception details may
                # reference in-memory inputs and are never logged.
                _transition(job, FAILED, ERR_PROOF_GENERATION_FAILED)
            finally:
                _release_features(job_id)
                features = None
        finally:
            _work_queue.task_done()


def _execute(job: _Job, features: list[float]) -> None:
    statement = job.statement
    backend = ensure_backend(statement["model_sha256"])
    compiled = str(backend / "model.compiled")
    srs = str(backend / "srs")
    q_score = statement["q_score"]

    input_bytes = json.dumps({"input_data": [features]}).encode("utf-8")
    with _memfd(input_bytes) as input_path:
        witness = ezkl.gen_witness(input_path, compiled, None)
    input_bytes = None
    outputs = witness.get("outputs") if isinstance(witness, dict) else None
    if not outputs or not outputs[0]:
        raise _ProofBackendError("witness missing public outputs")
    instance = outputs[0][0]
    felt = ezkl.felt_to_int(instance)
    if abs(felt - q_score) > INSTANCE_TOLERANCE:
        raise _ProofBackendError("public output inconsistent with statement")
    witness_bytes = json.dumps(witness).encode("utf-8")
    witness = None
    with _memfd(witness_bytes) as witness_path:
        proof = ezkl.prove(witness_path, compiled, str(backend / "pk"), None, srs)
    witness_bytes = None
    proof_hex = proof["proof"]
    proof_doc = {
        "instances": proof["instances"],
        "proof": list(bytes.fromhex(proof_hex[2:] if proof_hex.startswith("0x") else proof_hex)),
    }
    proof = None
    instances = proof_doc["instances"]
    _check_public_instances(instances)

    settings_bytes = (backend / "settings.json").read_bytes()
    vk_bytes = (backend / "vk").read_bytes()
    # Self-verify the fresh proof before marking the job succeeded.
    with _memfd(canonical_bytes(proof_doc)) as proof_path, _memfd(
        settings_bytes
    ) as settings_path, _memfd(vk_bytes) as vk_path:
        if not ezkl.verify(proof_path, settings_path, vk_path, srs):
            raise _ProofBackendError("fresh proof failed verification")

    directory = _job_dir(job.job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "proof.json").write_bytes(canonical_bytes(proof_doc))
    (directory / "settings.json").write_bytes(settings_bytes)
    (directory / "vk.bin").write_bytes(vk_bytes)
    (directory / "instances.json").write_bytes(canonical_bytes(instances))
    manifest = {
        "model_id": MODEL_ID,
        "model_sha256": statement["model_sha256"],
        "quantization_id": statement["quantization_id"],
        "q_score": q_score,
        "scale": statement["scale"],
        "rounding": statement["rounding"],
        "statement_sha256": statement["statement_sha256"],
        "ezkl_version": EZKL_VERSION,
        "materials": {
            "proof": _sha256_bytes((directory / "proof.json").read_bytes()),
            "settings": _sha256_bytes(settings_bytes),
            "vk": _sha256_bytes(vk_bytes),
            "instances": _sha256_bytes((directory / "instances.json").read_bytes()),
        },
    }
    (directory / "manifest.json").write_bytes(canonical_bytes(manifest))
    _transition(job, SUCCEEDED, None)


def _check_public_instances(instances) -> None:
    """Public instances must be exactly the single public model output."""
    if (
        not isinstance(instances, list)
        or len(instances) != 1
        or not isinstance(instances[0], list)
        or len(instances[0]) != 1
        or not _is_hex64(instances[0][0])
    ):
        raise _ProofBackendError("unexpected public instance shape")


# ---------------------------------------------------------------------------
# Proof material verification
# ---------------------------------------------------------------------------


def _expect(condition) -> None:
    if not condition:
        raise MaterialError()


def verify_materials(payload) -> bool:
    """Validate submitted proof material and run EZKL verification."""
    if ezkl is None:
        raise BackendUnavailable()
    material = _validate_material(payload)
    manifest = material["manifest"]
    # Model mapping is checked against the freshly audited release manifest.
    audited, _session = audited_artifacts(MODEL_ID)
    _expect(manifest["model_sha256"] == audited["sha256"])
    with _ezkl_lock:
        backend = ensure_backend(manifest["model_sha256"])
        proof_bytes = canonical_bytes(material["proof"])
        settings_bytes = canonical_bytes(material["settings"])
        vk_bytes = material["vk_bytes"]
        with _memfd(proof_bytes) as proof_path, _memfd(
            settings_bytes
        ) as settings_path, _memfd(vk_bytes) as vk_path:
            try:
                return bool(
                    ezkl.verify(proof_path, settings_path, vk_path, str(backend / "srs"))
                )
            except RuntimeError:
                raise MaterialError() from None


def _validate_material(payload) -> dict:
    """Structural, digest, and statement-mapping checks. Raises MaterialError."""
    _expect(isinstance(payload, dict))
    _expect(set(payload) == {"manifest", "proof", "settings", "instances", "vk"})
    manifest = payload["manifest"]
    proof = payload["proof"]
    settings = payload["settings"]
    instances = payload["instances"]
    vk = payload["vk"]

    # Manifest structure and model/statement mapping.
    _expect(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS)
    _expect(manifest["model_id"] == MODEL_ID)
    _expect(_is_hex64(manifest["model_sha256"]))
    _expect(manifest["quantization_id"] == QUANTIZATION_ID)
    _expect(manifest["scale"] == SCALE)
    _expect(manifest["rounding"] == ROUNDING)
    _expect(manifest["ezkl_version"] == EZKL_VERSION)
    q_score = manifest["q_score"]
    _expect(isinstance(q_score, int) and not isinstance(q_score, bool))
    _expect(0 <= q_score <= UINT32_MAX)
    _expect(_is_hex64(manifest["statement_sha256"]))
    digests = manifest["materials"]
    _expect(isinstance(digests, dict) and set(digests) == MATERIAL_DIGEST_KEYS)
    _expect(all(_is_hex64(digests[key]) for key in MATERIAL_DIGEST_KEYS))
    statement_body = {
        "model_id": manifest["model_id"],
        "model_sha256": manifest["model_sha256"],
        "quantization_id": manifest["quantization_id"],
        "q_score": q_score,
        "scale": manifest["scale"],
        "rounding": manifest["rounding"],
    }
    _expect(canonical_sha256(statement_body) == manifest["statement_sha256"])

    # Public instances: exactly the single public output, no private inputs.
    _expect(
        isinstance(instances, list)
        and len(instances) == 1
        and isinstance(instances[0], list)
        and len(instances[0]) == 1
        and _is_hex64(instances[0][0])
    )
    felt = ezkl.felt_to_int(instances[0][0])
    _expect(abs(felt - q_score) <= INSTANCE_TOLERANCE)

    # Proof document: canonical EZKL proof carrying the same public instances.
    _expect(isinstance(proof, dict) and set(proof) == {"instances", "proof"})
    _expect(proof["instances"] == instances)
    proof_bytes_list = proof["proof"]
    _expect(isinstance(proof_bytes_list, list) and len(proof_bytes_list) > 0)
    _expect(
        all(
            isinstance(byte, int) and not isinstance(byte, bool) and 0 <= byte <= 255
            for byte in proof_bytes_list
        )
    )

    # Settings must be a JSON object; the verification key is base64 bytes.
    _expect(isinstance(settings, dict) and settings)
    _expect(isinstance(vk, str) and vk)
    try:
        vk_bytes = base64.b64decode(vk, validate=True)
    except (binascii.Error, ValueError):
        raise MaterialError() from None
    _expect(len(vk_bytes) > 0)

    # Every material digest bound in the manifest must match.
    _expect(canonical_sha256(proof) == digests["proof"])
    _expect(canonical_sha256(settings) == digests["settings"])
    _expect(canonical_sha256(instances) == digests["instances"])
    _expect(_sha256_bytes(vk_bytes) == digests["vk"])

    return {
        "manifest": manifest,
        "proof": proof,
        "settings": settings,
        "instances": instances,
        "vk_bytes": vk_bytes,
    }
