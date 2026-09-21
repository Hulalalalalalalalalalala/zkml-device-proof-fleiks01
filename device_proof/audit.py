"""Fixed-model integrity audit for the bundled demonstration model.

The audit enforces the release contract for the single supported model
before any ordinary inference is served: the manifest must parse, have
correctly typed fields, and match the pinned release values; the ONNX
artifact must match the pinned SHA-256, pass the ONNX checker, and expose
exactly the expected inputs, outputs, operators, and initializers; and the
model must reproduce the pinned reference scores on CPU. The service only
ever runs ordinary inference; no zero-knowledge proofs are generated or
claimed anywhere in this audit.
"""

import copy
from hashlib import sha256
import json
from pathlib import Path
import threading

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "device-health-v1.onnx"
MANIFEST_PATH = ROOT / "models" / "device-health-v1.json"
MODEL_ID = "device-health-v1"

KNOWN_FEATURES = ("temperature", "vibration", "current", "runtime")
RELEASE_SHA256 = "4bf74b33ef32fe70aa57dcf5bd096b560d81af0c7eb0cb4378922f89ac66db8e"

EXPECTED_MANIFEST = {
    "id": MODEL_ID,
    "version": "1.0.0",
    "description": "Synthetic normalized device load score; not a diagnostic model.",
    "feature_order": ["temperature", "vibration", "current", "runtime"],
    "feature_range": [0.0, 1.0],
    "score_range": [0.1, 1.0],
    "score_direction": "higher-means-more-load",
    "input_shape": [1, 4],
    "output_shape": [1, 1],
    "onnx_opset": 13,
    "onnx_ir_version": 8,
    "sha256": RELEASE_SHA256,
}

EXPECTED_INITIALIZERS = {
    "weights": ([4, 1], [0.25, 0.25, 0.25, 0.15]),
    "bias": ([1], [0.1]),
}

REFERENCE_INPUTS = {
    "zeros": [0.0, 0.0, 0.0, 0.0],
    "sample": [0.2, 0.3, 0.4, 0.5],
    "ones": [1.0, 1.0, 1.0, 1.0],
}
REFERENCE_SCORES = {"zeros": 0.1, "sample": 0.4, "ones": 1.0}
REFERENCE_TOLERANCE = 1e-6

AUDIT_CHECKS = ["manifest", "sha256", "onnx_structure", "initializers", "reference_inference"]

_audit_lock = threading.Lock()
_session_lock = threading.Lock()
_sessions: dict = {}


class AuditError(Exception):
    """Audit failure carrying a stable, non-sensitive error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def audit_model(model_id: str = MODEL_ID) -> dict:
    """Run the full audit and return the public report. Raises AuditError."""
    report, _manifest, _session = _run_audit(model_id)
    return report


def audited_artifacts(model_id: str = MODEL_ID):
    """Run the full audit; return (manifest copy, inference session)."""
    _report, manifest, session = _run_audit(model_id)
    return manifest, session


def _run_audit(model_id: str):
    if model_id != MODEL_ID:
        raise AuditError("unknown_model")
    # Serialize audits so concurrent callers only ever observe complete results.
    with _audit_lock:
        manifest = _load_manifest()
        model_bytes = _read_bytes(MODEL_PATH, "model_missing")
        digest = sha256(model_bytes).hexdigest()
        if digest != manifest["sha256"]:
            raise AuditError("model_digest_mismatch")
        _check_onnx(model_bytes, manifest)
        session = _session_for(digest, model_bytes)
        scores = _reference_scores(session)
        report = {
            "model_id": manifest["id"],
            "version": manifest["version"],
            "status": "ok",
            "mode": "ordinary-inference",
            "sha256": digest,
            "onnx_opset": manifest["onnx_opset"],
            "onnx_ir_version": manifest["onnx_ir_version"],
            "checks": list(AUDIT_CHECKS),
            "reference_scores": scores,
        }
        return report, copy.deepcopy(manifest), session


def _read_bytes(path: Path, code: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise AuditError(code) from None


def _load_manifest() -> dict:
    raw = _read_bytes(MANIFEST_PATH, "manifest_missing")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise AuditError("manifest_invalid_json") from None
    return _validate_manifest(data)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_manifest(data) -> dict:
    if not isinstance(data, dict):
        raise AuditError("manifest_invalid_type")
    for key in ("id", "version", "description", "score_direction", "sha256"):
        if not isinstance(data.get(key), str):
            raise AuditError("manifest_invalid_type")
    feature_order = data.get("feature_order")
    if not isinstance(feature_order, list) or not all(
        isinstance(feature, str) for feature in feature_order
    ):
        raise AuditError("manifest_invalid_type")
    for key in ("feature_range", "score_range"):
        value = data.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(_is_number(bound) for bound in value)
        ):
            raise AuditError("manifest_invalid_type")
    for key in ("input_shape", "output_shape"):
        value = data.get(key)
        if (
            not isinstance(value, list)
            or not value
            or not all(_is_int(dim) for dim in value)
        ):
            raise AuditError("manifest_invalid_type")
    for key in ("onnx_opset", "onnx_ir_version"):
        if not _is_int(data.get(key)):
            raise AuditError("manifest_invalid_type")
    if len(set(feature_order)) != len(feature_order):
        raise AuditError("manifest_duplicate_feature")
    if any(feature not in KNOWN_FEATURES for feature in feature_order):
        raise AuditError("manifest_unknown_feature")
    for key, expected in EXPECTED_MANIFEST.items():
        if data.get(key) != expected:
            raise AuditError("manifest_contract_mismatch")
    return data


def _check_value_info(value_info, name: str, shape: list, code: str) -> None:
    type_proto = value_info.type
    if not type_proto.HasField("tensor_type"):
        raise AuditError(code)
    tensor = type_proto.tensor_type
    if (
        value_info.name != name
        or tensor.elem_type != onnx.TensorProto.FLOAT
        or not tensor.HasField("shape")
    ):
        raise AuditError(code)
    dims = []
    for dim in tensor.shape.dim:
        if not dim.HasField("dim_value"):
            raise AuditError(code)
        dims.append(dim.dim_value)
    if dims != list(shape):
        raise AuditError(code)


def _check_onnx(model_bytes: bytes, manifest: dict) -> None:
    try:
        model = onnx.load_model_from_string(model_bytes)
    except Exception:
        raise AuditError("model_unparseable") from None
    try:
        onnx.checker.check_model(model)
    except Exception:
        raise AuditError("model_checker_failed") from None
    if model.ir_version != manifest["onnx_ir_version"]:
        raise AuditError("model_ir_mismatch")
    opsets = [(opset.domain or "", opset.version) for opset in model.opset_import]
    if opsets != [("", manifest["onnx_opset"])]:
        raise AuditError("model_opset_mismatch")
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializer_names]
    if len(inputs) != 1:
        raise AuditError("model_input_mismatch")
    _check_value_info(inputs[0], "features", manifest["input_shape"], "model_input_mismatch")
    outputs = list(graph.output)
    if len(outputs) != 1:
        raise AuditError("model_output_mismatch")
    _check_value_info(outputs[0], "score", manifest["output_shape"], "model_output_mismatch")
    operators = sorted(node.op_type for node in graph.node)
    if operators != ["Add", "MatMul"] or any(
        node.domain not in ("", "ai.onnx") for node in graph.node
    ):
        raise AuditError("model_operator_mismatch")
    initializers = {init.name: init for init in graph.initializer}
    if set(initializers) != set(EXPECTED_INITIALIZERS):
        raise AuditError("model_initializer_mismatch")
    for name, (shape, values) in EXPECTED_INITIALIZERS.items():
        try:
            array = numpy_helper.to_array(initializers[name])
        except Exception:
            raise AuditError("model_initializer_mismatch") from None
        if (
            array.dtype != np.float32
            or list(array.shape) != list(shape)
            or not np.allclose(
                array.reshape(-1),
                np.asarray(values, dtype=np.float32),
                rtol=0,
                atol=REFERENCE_TOLERANCE,
            )
        ):
            raise AuditError("model_initializer_mismatch")


def _session_for(digest: str, model_bytes: bytes) -> ort.InferenceSession:
    # Keyed by the audited digest, so a file replaced in-process can never be
    # served from a stale cached session.
    with _session_lock:
        session = _sessions.get(digest)
        if session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            try:
                session = ort.InferenceSession(
                    model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
                )
            except Exception:
                raise AuditError("model_session_failed") from None
            _sessions.clear()
            _sessions[digest] = session
        return session


def _reference_scores(session: ort.InferenceSession) -> dict:
    scores = {}
    for label, values in REFERENCE_INPUTS.items():
        features = np.asarray([values], dtype=np.float32)
        try:
            output = session.run(["score"], {"features": features})[0]
        except Exception:
            raise AuditError("reference_inference_failed") from None
        scores[label] = float(output[0, 0])
    for label, expected in REFERENCE_SCORES.items():
        if abs(scores[label] - expected) > REFERENCE_TOLERANCE:
            raise AuditError("reference_inference_mismatch")
    return scores
