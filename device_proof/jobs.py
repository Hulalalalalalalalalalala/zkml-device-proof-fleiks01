"""Asynchronous local CPU proof jobs with a single FIFO worker.

Job lifecycle (terminal states never regress)::

    queued ──▶ running ──▶ succeeded
      │           │
      │           └────────▶ failed (proof_generation_failed)
      ├─▶ cancelled          (DELETE while queued)
      └────────▶ failed (interrupted) on restart
                  running ──▶ failed (interrupted) on restart

Persistence boundary
--------------------
Only non-sensitive metadata, the stable error code, and public proof
materials are persisted (under the runtime directory). Raw features,
per-feature encodings, and witnesses are never persisted: they live in
process memory and, for the duration of one EZKL invocation, in a
per-job tmpfs working directory that is wiped on any terminal state.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

from .audit import MODEL_ID, audited_artifacts
from .backend import EZKL_VERSION, EzklBackend, ProofBackendError
from .proof_materials import b64_encode, build_manifest
from .scoring import ROOT, ScoreRequest
from .statements import build_statement

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

_TERMINAL = {SUCCEEDED, FAILED, CANCELLED}
# Allowed forward transitions only.
_TRANSITIONS = {
    QUEUED: {RUNNING, SUCCEEDED, FAILED, CANCELLED},
    RUNNING: {SUCCEEDED, FAILED},
}

ERR_INTERRUPTED = "interrupted"
ERR_PROOF_FAILED = "proof_generation_failed"

# tmpfs for the sensitive per-job working directory; falls back to the
# platform temp dir if tmpfs is unavailable (the directory is still wiped
# the moment the job reaches a terminal state).
_TMPFS = Path("/dev/shm") if Path("/dev/shm").is_dir() else None


class JobError(Exception):
    def __init__(self, code: str, status_code: int):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def _feature_vector(request: ScoreRequest, feature_order: list) -> list:
    return [float(getattr(request.features, name)) for name in feature_order]


class JobStore:
    """File-backed, process-local job metadata and public materials."""

    def __init__(self, root: Path):
        self.root = root
        self.materials_dir = root / "materials"
        self.backend_dir = root / "backend"
        self._jobs_path = root / "jobs.json"
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        root.mkdir(parents=True, exist_ok=True)
        self.materials_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = self._jobs_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
            jobs = doc.get("jobs", {})
            if not isinstance(jobs, dict):
                raise ValueError
        except (OSError, ValueError):
            jobs = {}
        now = time.time()
        changed = False
        for record in jobs.values():
            # A process restart can never resume a job whose sensitive
            # inputs existed only in the previous process's memory.
            if record.get("status") in (QUEUED, RUNNING):
                record["status"] = FAILED
                record["error_code"] = ERR_INTERRUPTED
                record["finished_at"] = now
                changed = True
        self._jobs = jobs
        if changed:
            self._flush_locked()

    def _flush_locked(self) -> None:
        tmp = self._jobs_path.with_name("jobs.json.tmp")
        tmp.write_text(
            json.dumps({"jobs": self._jobs}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self._jobs_path)

    def _update_locked(self, job_id: str, **fields) -> dict:
        record = self._jobs[job_id]
        status = fields.get("status")
        if status is not None:
            allowed = _TRANSITIONS.get(record["status"], set())
            if status not in allowed:
                raise JobError("proof_job_terminal", 409)
            record["status"] = status
        for key, value in fields.items():
            record[key] = value
        self._flush_locked()
        return record

    # -- operations -----------------------------------------------------------

    def create(self, claim: dict) -> dict:
        job_id = uuid.uuid4().hex
        now = time.time()
        record = {
            "job_id": job_id,
            "status": QUEUED,
            "error_code": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            **claim,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._flush_locked()
        return dict(record)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def transition(self, job_id: str, **fields) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise JobError("proof_job_not_found", 404)
            return dict(self._update_locked(job_id, **fields))

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise JobError("proof_job_not_found", 404)
            if record["status"] == QUEUED:
                return dict(self._update_locked(
                    job_id, status=CANCELLED, finished_at=time.time()))
            # Running and every terminal state cannot be cancelled.
            raise JobError("proof_job_not_cancellable", 409)

    def status_of(self, job_id: str) -> str | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return record["status"] if record else None

    # -- public materials -----------------------------------------------------

    def save_materials(self, job_id: str, *, manifest: dict, proof: bytes,
                       settings: bytes, vk: bytes, instances) -> None:
        directory = self.materials_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)

        def atomic_write(name: str, data: bytes) -> None:
            target = directory / name
            tmp = directory / (name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)

        atomic_write("manifest.json",
                     json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        atomic_write("proof.json", proof)
        atomic_write("verification_key.key", vk)
        atomic_write("settings.json", settings)
        atomic_write("instances.json",
                     json.dumps(instances, ensure_ascii=False).encode("utf-8"))

    def load_materials(self, job_id: str) -> dict | None:
        directory = self.materials_dir / job_id
        try:
            manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
            proof = (directory / "proof.json").read_bytes()
            vk = (directory / "verification_key.key").read_bytes()
            settings = (directory / "settings.json").read_bytes()
            instances = json.loads((directory / "instances.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return {
            "manifest": manifest,
            "proof": proof,
            "verification_key": vk,
            "settings": settings,
            "instances": instances,
        }


def _new_private_dir(job_id: str) -> Path:
    # Sensitive features/encodings/witness only touch this memory-backed
    # directory, which the worker always removes before finishing.
    return Path(tempfile.mkdtemp(prefix=f"dp-proof-{job_id[:8]}-", dir=_TMPFS))


class JobManager:
    """Owns the store, the real EZKL backend, and one FIFO worker thread."""

    def __init__(self, runtime_root: Path | None = None,
                 backend_dir: Path | None = None):
        self.root = runtime_root or (ROOT / "runtime")
        self.store = JobStore(self.root)
        self._backend: EzklBackend | None = None
        self._backend_error: str | None = None
        self._backend_dir = backend_dir  # shared/prebuilt artifacts when given
        self._backend_lock = threading.Lock()
        self._queue: list[tuple[str, list]] = []
        self._condition = threading.Condition()
        self._stopped = False
        self._worker = threading.Thread(target=self._run_worker, name="proof-worker",
                                        daemon=True)
        self._worker.start()

    # -- backend lifecycle ----------------------------------------------------

    def backend(self) -> EzklBackend:
        """Return a ready backend or raise ProofBackendError(503 code)."""
        with self._backend_lock:
            if self._backend_error:
                raise ProofBackendError(self._backend_error)
            if self._backend is None:
                try:
                    backend = EzklBackend(
                        self._backend_dir or self.store.backend_dir)
                    # Provision the circuit before accepting jobs so a broken
                    # backend surfaces as 503 at submission, not mid-queue.
                    backend.trusted()
                except ProofBackendError as exc:
                    self._backend_error = exc.code
                    raise
                self._backend = backend
            return self._backend

    def ensure_backend(self) -> EzklBackend:
        """Return a backend instance if EZKL is present, without provisioning.

        Used to make "no backend → 503" win over request-body validation;
        the circuit itself is built lazily on the first real operation.
        """
        with self._backend_lock:
            if self._backend_error:
                raise ProofBackendError(self._backend_error)
            if self._backend is None:
                try:
                    self._backend = EzklBackend(
                        self._backend_dir or self.store.backend_dir)
                except ProofBackendError as exc:
                    self._backend_error = exc.code
                    raise
            return self._backend

    # -- submission -----------------------------------------------------------

    def submit(self, request: ScoreRequest) -> dict:
        if request.model_id != MODEL_ID:
            raise KeyError(request.model_id)
        manifest, _session = audited_artifacts(MODEL_ID)
        statement = build_statement(request)
        # A missing/broken EZKL backend is 503 before anything is queued.
        self.backend()

        claim = {
            "model_id": statement.model_id,
            "model_sha256": statement.model_sha256,
            "quantization_id": statement.quantization_id,
            "q_score": statement.q_score,
            "scale": statement.scale,
            "rounding": statement.rounding,
            "statement_sha256": statement.statement_sha256,
            "ezkl_version": EZKL_VERSION,
        }
        record = self.store.create(claim)
        features = _feature_vector(request, manifest["feature_order"])
        with self._condition:
            self._queue.append((record["job_id"], features))
            self._condition.notify()
        return record

    def cancel(self, job_id: str) -> dict:
        return self.store.cancel(job_id)

    def get(self, job_id: str) -> dict | None:
        record = self.store.get(job_id)
        if record is None:
            return None
        if record["status"] == SUCCEEDED:
            materials = self.store.load_materials(job_id)
            if materials is not None:
                record["materials"] = self._public_materials(materials)
        return record

    @staticmethod
    def _public_materials(materials: dict) -> dict:
        return {
            "manifest": materials["manifest"],
            "instances": materials["instances"],
            "proof": b64_encode(materials["proof"]),
            "verification_key": b64_encode(materials["verification_key"]),
            "settings": b64_encode(materials["settings"]),
        }

    # -- external verification ------------------------------------------------

    def verify_payload(self, payload: dict) -> dict:
        """Validate submitted material end to end and run real EZKL verify.

        Order: structure/digests (offline) → trusted model mapping → EZKL.
        Raises MaterialError on bad material and ProofBackendError when the
        local backend is unavailable.
        """
        from .proof_materials import (
            MaterialError, b64_decode, bind_materials, digest_bytes,
            settings_without_timestamp,
        )

        # Backend availability (503) is decided before body validation (422).
        backend = self.ensure_backend()

        for key in ("manifest", "proof", "verification_key", "settings"):
            if not isinstance(payload.get(key), str):
                raise MaterialError
        try:
            manifest = json.loads(payload["manifest"])
        except ValueError:
            raise MaterialError from None
        proof_bytes = b64_decode(payload["proof"])
        vk_bytes = b64_decode(payload["verification_key"])
        settings_bytes = b64_decode(payload["settings"])
        bound = bind_materials(
            manifest=manifest, proof_bytes=proof_bytes,
            settings_bytes=settings_bytes, vk_bytes=vk_bytes)

        # Model mapping: the claim must be for the audited release model and
        # the supplied verification key/settings must be the trusted circuit's
        # (the deterministic vk pins the exact ONNX/circuit the proof ran on;
        # settings are compared semantically because they carry a timestamp).
        trusted = self.backend().trusted()
        settings_match = (
            digest_bytes(settings_bytes) == trusted["settings_sha256"]
            or settings_without_timestamp(settings_bytes)
            == settings_without_timestamp(trusted["settings_bytes"])
        )
        if manifest["model_id"] != MODEL_ID \
                or manifest["model_sha256"] != trusted["model_sha256"] \
                or manifest["materials"]["verification_key_sha256"] != trusted["vk_sha256"] \
                or not settings_match:
            raise MaterialError

        try:
            verified = self.backend().verify_material(
                proof_bytes, settings_bytes, vk_bytes)
        except ProofBackendError as exc:
            # The backend uses its own exception type for a failed EZKL verify;
            # that means invalid material (422), not an unavailable backend.
            if exc.code == "invalid_proof_material":
                raise MaterialError from None
            raise
        if not verified:
            raise MaterialError
        return {
            "verified": True,
            "model_id": manifest["model_id"],
            "model_sha256": manifest["model_sha256"],
            "quantization_id": manifest["quantization_id"],
            "q_score": manifest["q_score"],
            "statement_sha256": manifest["statement_sha256"],
            "instances": bound["instances"],
            "ezkl_version": manifest["ezkl_version"],
        }

    # -- worker ---------------------------------------------------------------

    def _run_worker(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                item = self._queue.pop(0)
            if item is None:
                return  # stop sentinel
            job_id, features = item
            status = self.store.status_of(job_id)
            if status != QUEUED:
                # Cancelled (or otherwise moved on) while it waited: drop it.
                continue
            self._process(job_id, features)
            # Sensitive vector goes out of scope here; the private directory
            # is wiped inside _process on every outcome.
            del features

    def stop(self) -> None:
        """Stop the worker thread (used when replacing the manager in tests)."""
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            # Drop anything still queued (its sensitive vector dies with the
            # manager); a running job finishes on its own and then exits.
            self._queue.clear()
            self._queue.append(None)
            self._condition.notify()

    def _process(self, job_id: str, features: list) -> None:
        private_dir = _new_private_dir(job_id)
        try:
            self.store.transition(job_id, status=RUNNING, started_at=time.time())
            backend = self.backend()
            result = backend.prove(features, private_dir)
            record = self.store.get(job_id)
            manifest = build_manifest(
                model_sha256=result["model_sha256"],
                q_score=record["q_score"],
                statement_sha256=record["statement_sha256"],
                output_felt=result["output_felt"],
                proof_bytes=result["proof"],
                settings_bytes=result["settings"],
                vk_bytes=result["vk"],
                instances=result["instances"],
            )
            self.store.save_materials(
                job_id,
                manifest=manifest,
                proof=result["proof"],
                settings=result["settings"],
                vk=result["vk"],
                instances=result["instances"],
            )
            self.store.transition(job_id, status=SUCCEEDED, finished_at=time.time())
        except Exception:
            # Once queued, any runtime failure (EZKL error, backend lost after
            # submission, unexpected exception) is one stable code; internal
            # text is never persisted or surfaced.
            self._fail(job_id, ERR_PROOF_FAILED)
        finally:
            shutil.rmtree(private_dir, ignore_errors=True)

    def _fail(self, job_id: str, code: str) -> None:
        try:
            self.store.transition(job_id, status=FAILED, error_code=code,
                                  finished_at=time.time())
        except JobError:
            pass


_manager: "JobManager | None" = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_manager(runtime_root: Path | None = None,
                  backend_dir: Path | None = None) -> JobManager:
    """Test/operational hook: replace the process-wide manager."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop()
        _manager = JobManager(runtime_root, backend_dir=backend_dir)
        return _manager
