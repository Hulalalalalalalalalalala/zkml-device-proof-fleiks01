"""Fixed-model integrity audit and fail-closed access guard.

Only the released ``device-health-v1`` ordinary-inference model is supported.
There are no zero-knowledge proofs anywhere in this service: an audit verifies
that the on-disk manifest and ONNX artifact exactly match the published release
contract and that the artifact still produces the published golden outputs on
CPU.

The audit is a pure function over byte snapshots, so concurrent callers can
never observe a partially checked artifact. :class:`ModelGuard` revalidates
whenever either artifact changes, which means replacing a file in place always
takes effect immediately: a stale in-process session can never keep serving a
model that has been replaced on disk.
"""

from __future__ import annotations

from hashlib import sha256
import json
import threading
from pathlib import Path
from types import TracebackType

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort


# --- Stable error codes -----------------------------------------------------
# These tokens are the only thing emitted on failure paths (CLI stderr / API
# mapping). They must stay stable and must never embed paths or artifact data.

UNKNOWN_MODEL = "unknown_model"
ARTIFACT_MISSING = "artifact_missing"
MANIFEST_INVALID = "manifest_invalid"
MANIFEST_CONTRACT = "manifest_contract_violation"
MODEL_DIGEST_MISMATCH = "model_digest_mismatch"
MODEL_PARSE_FAILED = "model_parse_failed"
MODEL_CHECK_FAILED = "model_check_failed"
MODEL_CONTRACT_VIOLATED = "model_contract_violation"
MODEL_INFERENCE_FAILED = "model_inference_failed"
AUDIT_FAILED = "audit_failed"


class IntegrityError(Exception):
    """Raised when an artifact fails the published-contract audit."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


# --- Published release contract for device-health-v1 -----------------------

MODEL_ID = "device-health-v1"
MODEL_VERSION = "1.0.0"
FEATURE_ORDER = ("temperature", "vibration", "current", "runtime")
FEATURE_RANGE = (0.0, 1.0)
SCORE_RANGE = (0.1, 1.0)
INPUT_SHAPE = (1, 4)
OUTPUT_SHAPE = (1, 1)
ONNX_OPSET = 13
ONNX_IR_VERSION = 8
INPUT_NAME = "features"
OUTPUT_NAME = "score"
WEIGHTS_NAME = "weights"
BIAS_NAME = "bias"
WEIGHTS_SHAPE = (4, 1)
BIAS_SHAPE = (1,)
WEIGHTS_VALUES = (0.25, 0.25, 0.25, 0.15)
BIAS_VALUES = (0.1,)

# Golden CPU cases published with the release: zero input, the bundled
# example (examples/sample.json) and the all-ones boundary input.
GOLDEN_CASES = (
    ("zeros", (0.0, 0.0, 0.0, 0.0), 0.1),
    ("example", (0.2, 0.3, 0.4, 0.5), 0.4),
    ("ones", (1.0, 1.0, 1.0, 1.0), 1.0),
)
GOLDEN_TOLERANCE = 1e-6

_MANIFEST_FIELDS = {
    "id",
    "version",
    "description",
    "feature_order",
    "feature_range",
    "score_range",
    "score_direction",
    "input_shape",
    "output_shape",
    "onnx_opset",
    "onnx_ir_version",
    "sha256",
}


# --- Manifest validation ----------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_manifest(raw: object, expected_id: str) -> dict:
    if not isinstance(raw, dict):
        raise IntegrityError(MANIFEST_INVALID)
    if set(raw) != _MANIFEST_FIELDS:
        raise IntegrityError(MANIFEST_CONTRACT)

    def fail_invalid() -> None:
        raise IntegrityError(MANIFEST_INVALID)

    def fail_contract() -> None:
        raise IntegrityError(MANIFEST_CONTRACT)

    if not isinstance(raw["id"], str) or not isinstance(raw["version"], str):
        fail_invalid()
    if not isinstance(raw["description"], str) or not isinstance(
        raw["score_direction"], str
    ):
        fail_invalid()
    if not isinstance(raw["sha256"], str):
        fail_invalid()

    feature_order = raw["feature_order"]
    if not isinstance(feature_order, list) or not all(
        isinstance(name, str) for name in feature_order
    ):
        fail_invalid()
    if len(feature_order) != len(set(feature_order)):
        fail_contract()  # duplicated feature names
    if any(name not in FEATURE_ORDER for name in feature_order):
        fail_contract()  # unknown feature

    for key in ("feature_range", "score_range"):
        value = raw[key]
        if not isinstance(value, list) or len(value) != 2 or not all(
            _is_number(item) for item in value
        ):
            fail_invalid()

    for key in ("input_shape", "output_shape"):
        value = raw[key]
        if not isinstance(value, list) or not all(_is_int(item) for item in value):
            fail_invalid()

    if not _is_int(raw["onnx_opset"]) or not _is_int(raw["onnx_ir_version"]):
        fail_invalid()

    # Types are sound; every value must match the published contract.
    if raw["id"] != expected_id:
        fail_contract()
    if tuple(feature_order) != FEATURE_ORDER:
        fail_contract()
    if tuple(raw["feature_range"]) != FEATURE_RANGE:
        fail_contract()
    if tuple(raw["score_range"]) != SCORE_RANGE:
        fail_contract()
    if tuple(raw["input_shape"]) != INPUT_SHAPE:
        fail_contract()
    if tuple(raw["output_shape"]) != OUTPUT_SHAPE:
        fail_contract()
    if raw["onnx_opset"] != ONNX_OPSET:
        fail_contract()
    if raw["onnx_ir_version"] != ONNX_IR_VERSION:
        fail_contract()

    digest = raw["sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        fail_invalid()
    return raw


# --- ONNX structural validation --------------------------------------------


def _tensor_shape(value_info: onnx.TypeProto) -> list[int]:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            raise IntegrityError(MODEL_CONTRACT_VIOLATED)  # symbolic dimension
        dims.append(dim.dim_value)
    return dims


def _validate_graph(model: onnx.ModelProto) -> None:
    graph = model.graph

    inputs = list(graph.input)
    outputs = list(graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)

    feature_input = inputs[0]
    score_output = outputs[0]
    if feature_input.name != INPUT_NAME or score_output.name != OUTPUT_NAME:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    if feature_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    if score_output.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    if tuple(_tensor_shape(feature_input)) != INPUT_SHAPE:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    if tuple(_tensor_shape(score_output)) != OUTPUT_SHAPE:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)

    nodes = list(graph.node)
    expected_nodes = [
        ("MatMul", (INPUT_NAME, WEIGHTS_NAME), ("weighted",)),
        ("Add", ("weighted", BIAS_NAME), (OUTPUT_NAME,)),
    ]
    if len(nodes) != len(expected_nodes):
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    for node, (op_type, node_inputs, node_outputs) in zip(nodes, expected_nodes):
        if node.op_type != op_type:
            raise IntegrityError(MODEL_CONTRACT_VIOLATED)
        if tuple(node.input) != node_inputs or tuple(node.output) != node_outputs:
            raise IntegrityError(MODEL_CONTRACT_VIOLATED)

    initializers = {tensor.name: tensor for tensor in graph.initializer}
    if set(initializers) != {WEIGHTS_NAME, BIAS_NAME}:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)

    weights = initializers[WEIGHTS_NAME]
    bias = initializers[BIAS_NAME]
    if weights.data_type != TensorProto.FLOAT or bias.data_type != TensorProto.FLOAT:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    if tuple(weights.dims) != WEIGHTS_SHAPE or tuple(bias.dims) != BIAS_SHAPE:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    expected_weights = np.array(WEIGHTS_VALUES, dtype=np.float32).reshape(
        WEIGHTS_SHAPE
    )
    expected_bias = np.array(BIAS_VALUES, dtype=np.float32).reshape(BIAS_SHAPE)
    try:
        actual_weights = numpy_helper.to_array(weights).astype(np.float32, copy=False)
        actual_bias = numpy_helper.to_array(bias).astype(np.float32, copy=False)
    except Exception:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED) from None
    if not np.array_equal(actual_weights, expected_weights):
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    if not np.array_equal(actual_bias, expected_bias):
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)


# --- Snapshot audit ---------------------------------------------------------


def audit_snapshot(model_id: str, model_bytes: bytes, manifest_bytes: bytes) -> dict:
    """Audit byte snapshots of the manifest and ONNX artifact.

    Returns the single-line JSON-compatible audit report on success and raises
    :class:`IntegrityError` with a stable code on the first violation.
    """
    if model_id != MODEL_ID:
        raise IntegrityError(UNKNOWN_MODEL)

    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise IntegrityError(MANIFEST_INVALID) from None
    manifest = _validate_manifest(raw_manifest, model_id)

    digest = sha256(model_bytes).hexdigest()
    if manifest["sha256"] != digest:
        raise IntegrityError(MODEL_DIGEST_MISMATCH)

    try:
        model = onnx.load_from_string(model_bytes)
    except Exception:
        raise IntegrityError(MODEL_PARSE_FAILED) from None
    if model is None:
        raise IntegrityError(MODEL_PARSE_FAILED)

    try:
        onnx.checker.check_model(model)
    except Exception:
        raise IntegrityError(MODEL_CHECK_FAILED) from None

    if model.ir_version != manifest["onnx_ir_version"]:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)
    opsets = [(entry.domain or "", entry.version) for entry in model.opset_import]
    if opsets != [("", ONNX_OPSET)]:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED)

    try:
        _validate_graph(model)
    except IntegrityError:
        raise
    except Exception:
        raise IntegrityError(MODEL_CONTRACT_VIOLATED) from None

    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        for name, values, expected in GOLDEN_CASES:
            feed = np.array([values], dtype=np.float32)
            output = session.run([OUTPUT_NAME], {INPUT_NAME: feed})[0]
            actual = float(output[0, 0])
            if not abs(actual - expected) <= GOLDEN_TOLERANCE:
                raise IntegrityError(MODEL_INFERENCE_FAILED)
    except IntegrityError:
        raise
    except Exception:
        raise IntegrityError(MODEL_INFERENCE_FAILED) from None

    return build_report(manifest, digest)


def build_report(manifest: dict, digest: str) -> dict:
    """Assemble the stable audit report shared by the CLI and HTTP API."""
    return {
        "status": "verified",
        "model_id": manifest["id"],
        "version": manifest["version"],
        "mode": "ordinary-inference",
        "sha256": digest,
        "feature_order": list(manifest["feature_order"]),
        "feature_range": list(manifest["feature_range"]),
        "score_range": list(manifest["score_range"]),
        "input_shape": list(manifest["input_shape"]),
        "output_shape": list(manifest["output_shape"]),
        "onnx_opset": manifest["onnx_opset"],
        "onnx_ir_version": manifest["onnx_ir_version"],
        "inputs": [
            {"name": INPUT_NAME, "dtype": "FLOAT", "shape": list(INPUT_SHAPE)}
        ],
        "outputs": [
            {"name": OUTPUT_NAME, "dtype": "FLOAT", "shape": list(OUTPUT_SHAPE)}
        ],
        "operators": ["MatMul", "Add"],
        "initializers": [
            {"name": WEIGHTS_NAME, "dtype": "FLOAT", "shape": list(WEIGHTS_SHAPE)},
            {"name": BIAS_NAME, "dtype": "FLOAT", "shape": list(BIAS_SHAPE)},
        ],
        "golden_inference": [
            {"case": name, "expected": expected, "tolerance": GOLDEN_TOLERANCE}
            for name, _values, expected in GOLDEN_CASES
        ],
    }


# --- Fail-closed, replacement-sensitive process guard -----------------------


class _Entry:
    __slots__ = ("manifest", "model_digest", "manifest_digest", "session", "report")

    def __init__(
        self,
        manifest: dict,
        model_digest: str,
        manifest_digest: str,
        session: "ort.InferenceSession",
        report: dict,
    ) -> None:
        self.manifest = manifest
        self.model_digest = model_digest
        self.manifest_digest = manifest_digest
        self.session = session
        self.report = report


class ModelGuard:
    """Gatekeeper for the single supported model.

    Every access audits the current file contents against the cached entry's
    digests. Any byte-level change (including an in-place replacement that keeps
    the old file handle alive) invalidates the cache and is fully re-audited;
    an invalid replacement clears the cache and fails closed. Check and publish
    happen under one lock, so concurrent callers never see partial results.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._model_path = self._root / "models" / f"{MODEL_ID}.onnx"
        self._manifest_path = self._root / "models" / f"{MODEL_ID}.json"
        self._lock = threading.Lock()
        self._entry: _Entry | None = None

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def _read_snapshots_locked(self) -> tuple[bytes, bytes, str, str]:
        try:
            model_bytes = self._model_path.read_bytes()
            manifest_bytes = self._manifest_path.read_bytes()
        except OSError:
            self._entry = None
            raise IntegrityError(ARTIFACT_MISSING) from None
        return (
            model_bytes,
            manifest_bytes,
            sha256(model_bytes).hexdigest(),
            sha256(manifest_bytes).hexdigest(),
        )

    def access(self) -> _Entry:
        with self._lock:
            model_bytes, manifest_bytes, model_digest, manifest_digest = (
                self._read_snapshots_locked()
            )
            entry = self._entry
            if (
                entry is not None
                and entry.model_digest == model_digest
                and entry.manifest_digest == manifest_digest
            ):
                return entry

            try:
                report = audit_snapshot(MODEL_ID, model_bytes, manifest_bytes)
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                session = ort.InferenceSession(
                    model_bytes,
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                manifest = json.loads(manifest_bytes.decode("utf-8"))
            except IntegrityError:
                self._entry = None
                raise
            except Exception:
                self._entry = None
                raise IntegrityError(AUDIT_FAILED) from None

            self._entry = _Entry(
                manifest=manifest,
                model_digest=model_digest,
                manifest_digest=manifest_digest,
                session=session,
                report=report,
            )
            return self._entry

    # Context-manager support allows tests (and callers) to drop sessions.
    def __enter__(self) -> "ModelGuard":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        with self._lock:
            self._entry = None
