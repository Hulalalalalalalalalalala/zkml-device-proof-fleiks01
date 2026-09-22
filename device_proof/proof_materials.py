"""Proof material assembly and validation.

A successful proof job yields five non-sensitive, persistable materials:

* ``manifest``        – binds the claim (model, quantization, q_score,
                        statement digest) to every other material and to
                        EZKL 23.0.5 via SHA-256 digests;
* ``proof``           – the EZKL proof JSON (includes the public
                        ``instances``; blinding makes each proof unique);
* ``verification_key``– the deterministic EZKL verification key (binary);
* ``settings``        – the EZKL circuit settings JSON;
* ``instances``       – the public instances only (a single public output
                        felt); private inputs are never part of instances.

Raw features, per-feature encodings, and witnesses are never placed in any
material, manifest, response, log, or exception.
"""

import base64
from hashlib import sha256
import json

from .statements import QUANTIZATION_ID, ROUNDING, SCALE, _jcs_serialize
from .backend import EZKL_VERSION, PROOF_SCALE

MANIFEST_VERSION = "1.0"
# The public circuit output and q_score lie on the same Q16 grid; EZKL's
# re-quantization can shift the result by at most one fixed-point unit.
Q_SCORE_TOLERANCE = 1

_HEX_CHARS = set("0123456789abcdef")


class MaterialError(ValueError):
    """A material failed structural, digest, or model-mapping validation."""

    def __init__(self, code: str = "invalid_proof_material"):
        super().__init__(code)
        self.code = code


def digest_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def digest_json(value) -> str:
    """Digest over the RFC 8785 canonical form of a JSON value."""
    return sha256(_jcs_serialize(value).encode("utf-8")).hexdigest()


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise MaterialError
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception:
        raise MaterialError from None


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_CHARS for ch in value)
    )


def decode_output_felt(instances) -> int:
    if (
        not isinstance(instances, list)
        or len(instances) != 1
        or not isinstance(instances[0], list)
        or len(instances[0]) != 1
        or not _is_hex64(instances[0][0])
    ):
        raise MaterialError
    return int.from_bytes(bytes.fromhex(instances[0][0])[::-1], "big")


def parse_proof(proof_bytes: bytes) -> dict:
    """Parse and structurally validate an EZKL proof JSON document."""
    try:
        doc = json.loads(proof_bytes)
    except (UnicodeDecodeError, ValueError):
        raise MaterialError from None
    if not isinstance(doc, dict):
        raise MaterialError
    if not isinstance(doc.get("proof"), list) or not doc["proof"]:
        raise MaterialError
    if not all(isinstance(byte, int) and 0 <= byte <= 255
               for byte in doc["proof"]):
        raise MaterialError
    if not isinstance(doc.get("hex_proof"), str) or not doc["hex_proof"]:
        raise MaterialError
    if doc.get("version") != EZKL_VERSION:
        raise MaterialError
    instances = doc.get("instances")
    # Shape-only validation here; felt decoding is done separately.
    decode_output_felt(instances)
    return doc


def parse_settings(settings_bytes: bytes) -> dict:
    """Parse and structurally validate EZKL settings, returning normalized copy."""
    try:
        doc = json.loads(settings_bytes)
    except (UnicodeDecodeError, ValueError):
        raise MaterialError from None
    if not isinstance(doc, dict) or not isinstance(doc.get("run_args"), dict):
        raise MaterialError
    run = doc["run_args"]
    if (
        run.get("input_visibility") != "Private"
        or run.get("output_visibility") != "Public"
        or doc.get("model_input_scales") != [PROOF_SCALE]
        or doc.get("model_output_scales") != [PROOF_SCALE]
        or doc.get("model_instance_shapes") != [[1, 1]]
        or doc.get("version") != EZKL_VERSION
    ):
        raise MaterialError
    normalized = dict(doc)
    # The timestamp is the one per-build, non-semantic settings field.
    normalized.pop("timestamp", None)
    return normalized


def build_manifest(*, model_sha256: str, q_score: int, statement_sha256: str,
                   output_felt: int, proof_bytes: bytes, settings_bytes: bytes,
                   vk_bytes: bytes, instances) -> dict:
    """Assemble the digest-binding manifest for a freshly generated proof."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "model_id": "device-health-v1",
        "model_sha256": model_sha256,
        "quantization_id": QUANTIZATION_ID,
        "q_score": q_score,
        "scale": SCALE,
        "rounding": ROUNDING,
        "statement_sha256": statement_sha256,
        "ezkl_version": EZKL_VERSION,
        "circuit": {
            "input_visibility": "Private",
            "output_visibility": "Public",
            "output_scale": PROOF_SCALE,
        },
        "public_output": {"felt": output_felt, "scale": PROOF_SCALE},
        "instances_sha256": digest_json(instances),
        "materials": {
            "proof_sha256": digest_bytes(proof_bytes),
            "verification_key_sha256": digest_bytes(vk_bytes),
            "settings_sha256": digest_bytes(settings_bytes),
        },
    }


def statement_digest_from_manifest(manifest: dict) -> str:
    """Recompute the statement digest bound inside a manifest.

    Mirrors statements.build_statement: the digest preimage is exactly the
    claim fields, JCS-canonicalized.
    """
    body = {
        "model_id": manifest["model_id"],
        "model_sha256": manifest["model_sha256"],
        "quantization_id": manifest["quantization_id"],
        "q_score": manifest["q_score"],
        "scale": manifest["scale"],
        "rounding": manifest["rounding"],
    }
    return digest_json(body)


def validate_manifest_shape(manifest, *, pin_model: bool = True) -> dict:
    """Type-check the manifest enough to safely read its binding fields.

    With ``pin_model`` (the default) the model/quantization identity fields
    are also pinned to the one audited release scheme; bundle verification
    passes ``pin_model=False`` so a model/quantization conflict can surface
    as its own distinct error instead of a material error.
    """
    if not isinstance(manifest, dict):
        raise MaterialError
    required_str = {
        "manifest_version", "model_id", "model_sha256", "quantization_id",
        "statement_sha256", "ezkl_version",
    }
    for key in required_str:
        if not isinstance(manifest.get(key), str):
            raise MaterialError
    for key in ("q_score", "scale"):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise MaterialError
    if not isinstance(manifest.get("rounding"), str):
        raise MaterialError
    circuit = manifest.get("circuit")
    output = manifest.get("public_output")
    materials = manifest.get("materials")
    if not isinstance(circuit, dict) or not isinstance(output, dict) \
            or not isinstance(materials, dict):
        raise MaterialError
    if (
        circuit.get("input_visibility") != "Private"
        or circuit.get("output_visibility") != "Public"
        or circuit.get("output_scale") != PROOF_SCALE
        or output.get("scale") != PROOF_SCALE
        or not isinstance(output.get("felt"), int)
    ):
        raise MaterialError
    for key in ("proof_sha256", "verification_key_sha256", "settings_sha256"):
        if not _is_hex64(materials.get(key)):
            raise MaterialError
    if not _is_hex64(manifest.get("instances_sha256")):
        raise MaterialError
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise MaterialError
    if manifest["ezkl_version"] != EZKL_VERSION:
        raise MaterialError
    if pin_model:
        # The semantic interpretation of the public output is pinned to the
        # one quantization scheme the circuit was built for; a claimant cannot
        # relabel a genuine proof as belonging to a different scheme (even
        # though it could recompute the self-consistent statement digest).
        if manifest["model_id"] != "device-health-v1":
            raise MaterialError
        if manifest["quantization_id"] != QUANTIZATION_ID:
            raise MaterialError
        if manifest["scale"] != SCALE or manifest["rounding"] != ROUNDING:
            raise MaterialError
    if not _is_hex64(manifest["model_sha256"]):
        raise MaterialError
    return manifest


def bind_materials(*, manifest: dict, proof_bytes: bytes, settings_bytes: bytes,
                   vk_bytes: bytes, pin_model: bool = True) -> dict:
    """Validate structure and all digest bindings of supplied materials.

    Returns ``{"instances": ..., "output_felt": ..., "settings_normalized": ...}``
    on success; raises MaterialError("invalid_proof_material") otherwise.
    Cryptographic EZKL verification and the trusted model mapping happen in
    the caller (they need the backend). ``pin_model`` is forwarded to
    validate_manifest_shape.
    """
    validate_manifest_shape(manifest, pin_model=pin_model)
    parse_settings(settings_bytes)  # structural only
    proof_doc = parse_proof(proof_bytes)
    instances = proof_doc["instances"]
    output_felt = decode_output_felt(instances)

    materials = manifest["materials"]
    if materials["proof_sha256"] != digest_bytes(proof_bytes):
        raise MaterialError
    if materials["settings_sha256"] != digest_bytes(settings_bytes):
        raise MaterialError
    if materials["verification_key_sha256"] != digest_bytes(vk_bytes):
        raise MaterialError
    if manifest["instances_sha256"] != digest_json(instances):
        raise MaterialError
    if output_felt != manifest["public_output"]["felt"]:
        raise MaterialError
    if abs(output_felt - manifest["q_score"]) > Q_SCORE_TOLERANCE:
        raise MaterialError
    if statement_digest_from_manifest(manifest) != manifest["statement_sha256"]:
        raise MaterialError
    settings_normalized = parse_settings(settings_bytes)
    return {
        "instances": instances,
        "output_felt": output_felt,
        "settings_normalized": settings_normalized,
    }


def settings_without_timestamp(settings_bytes: bytes) -> dict:
    """Normalized (timestamp-free) settings for the trusted model mapping."""
    return parse_settings(settings_bytes)
