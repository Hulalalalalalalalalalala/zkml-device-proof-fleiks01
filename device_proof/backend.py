"""Local CPU proving backend backed by the real EZKL 23.0.5 pipeline.

Every public operation here drives the genuine EZKL toolchain
(``gen_settings`` → ``calibrate_settings`` → ``compile_circuit`` →
``get_srs`` → ``setup`` → ``gen_witness`` → ``prove`` / ``verify``).
Nothing is mocked, stubbed, or short-circuited: a returned proof was
produced by ``ezkl.prove`` on CPU and a positive verification was
produced by ``ezkl.verify``.

Privacy boundary
----------------
Raw features, their field encodings, and the witness are sensitive. They
exist only inside a per-job temporary working directory for the duration
of a single proof attempt and are wiped (files unlinked, directory
removed) as soon as the job reaches a terminal state. The calibration
data the circuit is built from is fixed public data (all-zero, all-one,
and the release sample); it never contains a submitted feature vector.

The reusable circuit artifacts (settings, compiled circuit, verification
key, proving key, SRS) are non-sensitive public material and are cached
in a backend-private working directory.
"""

import asyncio
import copy
from hashlib import sha256
import inspect
import json
import shutil
import tempfile
import threading
from pathlib import Path

from .audit import MODEL_ID, MODEL_PATH, audited_artifacts

EZKL_VERSION = "23.0.5"

# The circuit is deliberately built at scale 2**16 so it lines up with the
# Q16 quantization used by statements.device-health-v1-q16-v1: the public
# circuit output is then on the same integer grid as q_score (within one
# LSB of re-quantization noise).
PROOF_SCALE = 16
PROOF_LOGROWS = 15

# Fixed, public calibration points keyed by feature name: the score-range
# endpoints and the release sample. No submitted feature vector is ever
# used to build or calibrate the circuit.
_CALIBRATION_POINTS = [
    {"temperature": 0.0, "vibration": 0.0, "current": 0.0, "runtime": 0.0},
    {"temperature": 1.0, "vibration": 1.0, "current": 1.0, "runtime": 1.0},
    {"temperature": 0.2, "vibration": 0.3, "current": 0.4, "runtime": 0.5},
]


class ProofBackendError(Exception):
    """Backend failure carrying a stable, non-sensitive error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def _decode_output_felt(instances) -> int | None:
    """Decode the single public output felt from EZKL instances (LE bytes)."""
    if (
        not isinstance(instances, list)
        or len(instances) != 1
        or not isinstance(instances[0], list)
        or len(instances[0]) != 1
        or not isinstance(instances[0][0], str)
        or len(instances[0][0]) != 64
    ):
        return None
    try:
        raw = bytes.fromhex(instances[0][0])
    except ValueError:
        return None
    return int.from_bytes(raw[::-1], "big")


class _LoopThread:
    """Run EZKL's async pyo3 bindings on one persistent event-loop thread.

    EZKL's async functions require a running event loop bound to the
    calling thread, so all invocations are marshalled here. A single loop
    also serializes access to EZKL's global table/cache state.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="ezkl-loop", daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def call(self, fn):
        async def _drive():
            result = fn()
            while inspect.isawaitable(result):
                result = await result
            return result

        future = asyncio.run_coroutine_threadsafe(_drive(), self._loop)
        return future.result()


_LOOP: "_LoopThread | None" = None
_LOOP_LOCK = threading.Lock()


def _loop_thread() -> _LoopThread:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = _LoopThread()
        return _LOOP


class EzklBackend:
    """Real EZKL proving backend for the single audited release model."""

    def __init__(self, workdir: Path):
        try:
            import ezkl  # imported lazily so a missing wheel degrades to 503
        except Exception as exc:
            raise ProofBackendError("proof_backend_unavailable") from exc
        if getattr(ezkl, "__version__", EZKL_VERSION) != EZKL_VERSION:
            raise ProofBackendError("proof_backend_unavailable")
        self._ezkl = ezkl
        self._workdir = workdir
        self._ready = False
        self._init_lock = threading.Lock()
        self._loop = _loop_thread()
        # Public, reusable material, populated by _build().
        self._model_sha256 = None
        self._settings_bytes = None
        self._vk_bytes = None
        self._compiled_path = None
        self._pk_path = None
        self._srs_path = None

    # -- circuit provisioning -------------------------------------------------

    def _artifact_bundle(self) -> dict:
        with self._init_lock:
            if not self._ready:
                try:
                    if not self._load_cache():
                        self._build()
                        self._selftest()
                except ProofBackendError:
                    raise
                except Exception:
                    # Never leak EZKL/Rust text or artifact internals.
                    raise ProofBackendError("proof_backend_unavailable") from None
                self._ready = True
        return {
            "model_sha256": self._model_sha256,
            "settings": self._settings_bytes,
            "vk": self._vk_bytes,
            "compiled_path": self._compiled_path,
            "pk_path": self._pk_path,
            "srs_path": self._srs_path,
        }

    def _settings_ok(self, settings_path: Path) -> bool:
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        run = settings.get("run_args", {})
        return (
            run.get("input_visibility") == "Private"
            and run.get("output_visibility") == "Public"
            and settings.get("model_input_scales") == [PROOF_SCALE]
            and settings.get("model_output_scales") == [PROOF_SCALE]
            and settings.get("version") == EZKL_VERSION
        )

    def _load_cache(self) -> bool:
        """Reuse previously built public artifacts when they still verify.

        Returns True only when every artifact is present and structurally
        valid AND a persisted real proof verifies against them, which proves
        the circuit/vk/pk/srs set is intact and mutually consistent.
        """
        work = self._workdir
        paths = {
            "meta": work / "build.json",
            "settings": work / "settings.json",
            "compiled": work / "circuit.ezkl",
            "srs": work / "kzg.srs",
            "vk": work / "vk.key",
            "pk": work / "pk.key",
            "selftest_proof": work / "selftest" / "proof.json",
        }
        if not all(path.exists() for path in paths.values()):
            return False
        if not self._settings_ok(paths["settings"]):
            return False
        manifest, _session = audited_artifacts(MODEL_ID)
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if meta.get("model_sha256") != manifest["sha256"] \
                or meta.get("ezkl_version") != EZKL_VERSION:
            return False
        try:
            ok = bool(self._loop.call(lambda: self._ezkl.verify(
                str(paths["selftest_proof"]), str(paths["settings"]),
                str(paths["vk"]), srs_path=str(paths["srs"]))))
        except Exception:
            return False
        if not ok:
            return False
        self._model_sha256 = manifest["sha256"]
        self._settings_bytes = paths["settings"].read_bytes()
        self._vk_bytes = paths["vk"].read_bytes()
        self._compiled_path = paths["compiled"]
        self._pk_path = paths["pk"]
        self._srs_path = paths["srs"]
        return True

    def _build(self) -> None:
        manifest, _session = audited_artifacts(MODEL_ID)
        feature_order = manifest["feature_order"]
        # Read the audited bytes straight from the release artifact so the
        # circuit is provably built from exactly the bytes the audit pinned.
        model_bytes = MODEL_PATH.read_bytes()
        if sha256(model_bytes).hexdigest() != manifest["sha256"]:
            raise ProofBackendError("proof_backend_unavailable")

        work = self._workdir
        work.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(work / "selftest", ignore_errors=True)
        model_path = work / "model.onnx"
        settings_path = work / "settings.json"
        compiled_path = work / "circuit.ezkl"
        srs_path = work / "kzg.srs"
        vk_path = work / "vk.key"
        pk_path = work / "pk.key"
        calib_path = work / "calibration.json"
        model_path.write_bytes(model_bytes)
        calibration = {
            "input_data": [[point[name] for name in feature_order]
                           for point in _CALIBRATION_POINTS]
        }
        calib_path.write_text(json.dumps(calibration), encoding="utf-8")

        run_args = self._ezkl.PyRunArgs()
        run_args.input_visibility = "Private"
        run_args.output_visibility = "Public"
        run_args.param_visibility = "Fixed"
        run_args.input_scale = PROOF_SCALE
        run_args.param_scale = PROOF_SCALE
        run_args.logrows = PROOF_LOGROWS
        self._loop.call(lambda: self._ezkl.gen_settings(
            str(model_path), str(settings_path), run_args))
        self._loop.call(lambda: self._ezkl.calibrate_settings(
            str(calib_path), str(model_path), str(settings_path),
            "resources", scales=[PROOF_SCALE]))
        self._loop.call(lambda: self._ezkl.compile_circuit(
            str(model_path), str(compiled_path), str(settings_path)))

        if not self._settings_ok(settings_path):
            raise ProofBackendError("proof_backend_unavailable")

        self._provision_srs(settings_path, srs_path)
        self._loop.call(lambda: self._ezkl.setup(
            str(compiled_path), str(vk_path), str(pk_path), srs_path=str(srs_path)))

        self._model_sha256 = manifest["sha256"]
        self._settings_bytes = settings_path.read_bytes()
        self._vk_bytes = vk_path.read_bytes()
        self._compiled_path = compiled_path
        self._pk_path = pk_path
        self._srs_path = srs_path
        (work / "build.json").write_text(
            json.dumps({
                "model_sha256": manifest["sha256"],
                "ezkl_version": EZKL_VERSION,
            }, sort_keys=True),
            encoding="utf-8",
        )

    def _provision_srs(self, settings_path: Path, srs_path: Path) -> None:
        """Provide the deterministic public KZG SRS, offline when possible.

        The SRS is a fixed public-ceremony file keyed by logrows. Reuse a copy
        already present in EZKL's standard cache when its size matches, then
        always defer to get_srs to validate it (or fetch it if absent); this
        avoids a network dependency on first build while keeping integrity.
        """
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            logrows = int(settings["run_args"]["logrows"])
            cached = Path.home() / ".ezkl" / "srs" / f"kzg{logrows}.srs"
            if cached.exists() and not srs_path.exists():
                shutil.copyfile(cached, srs_path)
        except (OSError, ValueError, KeyError):
            pass
        self._loop.call(lambda: self._ezkl.get_srs(
            str(settings_path), srs_path=str(srs_path)))

    def _selftest(self) -> None:
        """Generate and persist one real proof over the public zero input.

        It is a genuine EZKL proof (never a mock) over fixed public data; on
        later process starts it is verified to prove the cached artifacts are
        intact and mutually consistent before they are reused.
        """
        work = self._workdir
        check_dir = work / "selftest"
        check_dir.mkdir(parents=True, exist_ok=True)
        data_path = check_dir / "input.json"
        witness_path = check_dir / "witness.json"
        proof_path = check_dir / "proof.json"
        data_path.write_text(
            json.dumps({"input_data": [[0.0, 0.0, 0.0, 0.0]]}),
            encoding="utf-8",
        )
        try:
            self._loop.call(lambda: self._ezkl.gen_witness(
                str(data_path), str(self._compiled_path), str(witness_path)))
            self._loop.call(lambda: self._ezkl.prove(
                str(witness_path), str(self._compiled_path), str(self._pk_path),
                proof_path=str(proof_path), srs_path=str(self._srs_path)))
            ok = self._loop.call(lambda: self._ezkl.verify(
                str(proof_path), str(work / "settings.json"),
                str(work / "vk.key"), srs_path=str(self._srs_path)))
            if not ok:
                raise ProofBackendError("proof_backend_unavailable")
        except ProofBackendError:
            raise
        except Exception:
            raise ProofBackendError("proof_backend_unavailable") from None
        finally:
            # The self-test witness would only encode public zeros, but wipe
            # it anyway so no witness is ever left on disk.
            witness_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)

    # -- proving --------------------------------------------------------------

    def prove(self, features: list, job_dir: Path) -> dict:
        """Run a real proof for one feature vector ordered by feature_order.

        Returns public material only: settings/vk/proof bytes, the public
        instances, and the output felt. Sensitive inputs and the witness
        live solely in ``job_dir``; the caller wipes that directory when the
        job reaches a terminal state.
        """
        bundle = self._artifact_bundle()
        job_dir.mkdir(parents=True, exist_ok=True)
        data_path = job_dir / "input.json"
        witness_path = job_dir / "witness.json"
        proof_path = job_dir / "proof.json"
        # Only the sensitive input/witness and the produced proof live in the
        # per-job directory. The proving key is non-sensitive public material
        # and read-only, so it is used straight from the cache (no large copy).
        data_path.write_text(json.dumps({"input_data": [features]}), encoding="utf-8")
        try:
            witness = self._loop.call(lambda: self._ezkl.gen_witness(
                str(data_path), str(bundle["compiled_path"]), str(witness_path)))
            # Visibility is enforced at circuit build time; this shape check is
            # defense in depth so a single public output is the only thing we
            # ever accept.
            public_outputs = witness.get("outputs") or []
            if len(public_outputs) != 1 or len(public_outputs[0]) != 1:
                raise ProofBackendError("proof_generation_failed")
            self._loop.call(lambda: self._ezkl.prove(
                str(witness_path), str(bundle["compiled_path"]),
                str(bundle["pk_path"]),
                proof_path=str(proof_path), srs_path=str(bundle["srs_path"])))
            proof_bytes = proof_path.read_bytes()
            proof_doc = json.loads(proof_bytes)
        except ProofBackendError:
            raise
        except Exception:
            raise ProofBackendError("proof_generation_failed") from None

        instances = proof_doc.get("instances")
        output_felt = _decode_output_felt(instances)
        if output_felt is None:
            raise ProofBackendError("proof_generation_failed")
        # A fresh proof must verify against the trusted verification key.
        self._verify_with_trusted(proof_bytes, bundle["settings"], bundle["vk"])
        return {
            "settings": bundle["settings"],
            "vk": bundle["vk"],
            "proof": proof_bytes,
            "instances": copy.deepcopy(instances),
            "output_felt": output_felt,
            "model_sha256": bundle["model_sha256"],
        }

    # -- verification ---------------------------------------------------------

    def verify_material(self, proof_bytes: bytes, settings_bytes: bytes,
                        vk_bytes: bytes) -> bool:
        """Verify supplied public material with the real EZKL verifier.

        Structural and digest validation happens in the caller; here EZKL
        either returns True or raises. Any failure is invalid material.
        """
        self._artifact_bundle()
        tmp = Path(tempfile.mkdtemp(prefix="ezkl-verify-"))
        try:
            (tmp / "proof.json").write_bytes(proof_bytes)
            (tmp / "settings.json").write_bytes(settings_bytes)
            (tmp / "vk.key").write_bytes(vk_bytes)
            try:
                return bool(self._loop.call(lambda: self._ezkl.verify(
                    str(tmp / "proof.json"), str(tmp / "settings.json"),
                    str(tmp / "vk.key"), srs_path=str(self._srs_path))))
            except Exception:
                raise ProofBackendError("invalid_proof_material") from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _verify_with_trusted(self, proof_bytes: bytes, settings_bytes: bytes,
                             vk_bytes: bytes) -> None:
        try:
            ok = self.verify_material(proof_bytes, settings_bytes, vk_bytes)
        except ProofBackendError as exc:
            if exc.code == "invalid_proof_material":
                raise ProofBackendError("proof_generation_failed") from None
            raise
        if not ok:
            raise ProofBackendError("proof_generation_failed")

    # -- trusted references ---------------------------------------------------

    def trusted(self) -> dict:
        """Trusted references used for digest checks and model mapping."""
        bundle = self._artifact_bundle()
        return {
            "model_sha256": bundle["model_sha256"],
            "settings_sha256": _sha256_hex(bundle["settings"]),
            "vk_sha256": _sha256_hex(bundle["vk"]),
            "settings_bytes": bundle["settings"],
            "vk_bytes": bundle["vk"],
        }
