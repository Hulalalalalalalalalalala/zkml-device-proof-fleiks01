"""Versioned evidence bundles: export and standalone verification.

An evidence bundle packages the public proof materials of one succeeded
job as a single zip whose root contains exactly::

    bundle.json            – bundle_version plus the SHA-256 of every
                             other member (the bundle's own integrity map)
    manifest.json          – the digest-binding claim manifest
    proof.json             – the EZKL proof (public instances only)
    verification_key.key   – the deterministic EZKL verification key
    settings.json          – the EZKL circuit settings
    instances.json         – the public instances (one output felt)

Raw features, per-feature encodings, witnesses, and internal filesystem
paths are never part of a bundle: the member set is fixed, the names are
relative, and the contents are exactly the persisted public materials.

Verification is fully standalone: it never starts the HTTP service and
never opens the runtime job store. It rejects hostile archives (absolute
paths, ``..`` components, duplicate names, directories, links, encrypted
or oversized members) before any content is parsed, then checks the
bundle version and member digests, the manifest/instances binding, the
trusted model and quantization mapping, and finally runs the real EZKL
verifier.
"""

import json
import os
from pathlib import Path
import zipfile

from .audit import MODEL_ID
from .backend import EzklBackend, ProofBackendError
from .jobs import SUCCEEDED, JobStore
from .proof_materials import (
    MaterialError,
    decode_output_felt,
    digest_bytes,
    digest_json,
    parse_proof,
    parse_settings,
    settings_without_timestamp,
    statement_digest_from_manifest,
    validate_manifest_structure,
    Q_SCORE_TOLERANCE,
)
from .scoring import ROOT
from .statements import QUANTIZATION_ID, ROUNDING, SCALE

BUNDLE_VERSION = 1

# Fixed root member set, in canonical write order.
BUNDLE_MEMBERS = (
    "bundle.json",
    "manifest.json",
    "proof.json",
    "verification_key.key",
    "settings.json",
    "instances.json",
)
# Members whose digests are pinned inside bundle.json.
_DIGESTED_MEMBERS = BUNDLE_MEMBERS[1:]

# Archive safety limits: the fixed member set is tiny, and the genuine
# materials are well under a megabyte in total, so these bounds are
# generous while still rejecting member floods and compression bombs.
MAX_BUNDLE_MEMBERS = len(BUNDLE_MEMBERS)
MAX_BUNDLE_UNCOMPRESSED = 64 * 1024 * 1024

# Regular-file type bits in the high word of external attributes.
_S_IFMT = 0o170000
_S_IFREG = 0o100000


class BundleError(Exception):
    """Bundle export/verification failure carrying a stable error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# -- export -------------------------------------------------------------------


def export_bundle(job_id: str, output: Path, runtime_root: Path) -> None:
    """Write the evidence bundle for one succeeded job to ``output``.

    Raises BundleError("proof_job_not_found") for an unknown job and
    BundleError("proof_bundle_not_ready") for any non-succeeded state.
    """
    store = JobStore(runtime_root)
    record = store.get(job_id)
    if record is None:
        raise BundleError("proof_job_not_found")
    if record["status"] != SUCCEEDED:
        raise BundleError("proof_bundle_not_ready")
    materials = store.load_material_bytes(job_id)
    if materials is None:
        # A succeeded job without its full public material set is not
        # exportable; nothing about the internals is surfaced.
        raise BundleError("proof_bundle_not_ready")

    digests = {name: digest_bytes(materials[name]) for name in _DIGESTED_MEMBERS}
    bundle_doc = json.dumps(
        {"bundle_version": BUNDLE_VERSION, "files": digests},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    files = {"bundle.json": bundle_doc, **materials}

    tmp = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            for name in BUNDLE_MEMBERS:
                # Fixed timestamp and plain regular-file attributes keep the
                # archive reproducible and free of host metadata.
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (_S_IFREG | 0o644) << 16
                zf.writestr(info, files[name])
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# -- archive safety -----------------------------------------------------------


def _check_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    # Directories and non-regular entries (symlinks, devices, ...) are never
    # legitimate bundle members.
    if info.is_dir():
        raise BundleError("invalid_bundle")
    mode = (info.external_attr >> 16) & _S_IFMT
    if mode and mode != _S_IFREG:
        raise BundleError("invalid_bundle")
    if info.flag_bits & 0x1:  # encrypted member
        raise BundleError("invalid_bundle")
    # Only plain relative names: no absolute paths, drive letters,
    # backslashes, or dot segments.
    if not name or name.startswith("/") or "\\" in name:
        raise BundleError("invalid_bundle")
    if len(name) > 1 and name[1] == ":":
        raise BundleError("invalid_bundle")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise BundleError("invalid_bundle")


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, budget: int) -> bytes:
    """Read one member, capping the expanded size at the remaining budget."""
    try:
        with zf.open(info) as handle:
            data = handle.read(budget + 1)
    except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile):
        raise BundleError("invalid_bundle") from None
    if len(data) > budget:
        raise BundleError("invalid_bundle")
    return data


def _read_archive(bundle: Path) -> dict:
    """Open the bundle and return {name: bytes} after all safety checks."""
    try:
        zf = zipfile.ZipFile(bundle)
    except (OSError, zipfile.BadZipFile):
        raise BundleError("invalid_bundle") from None
    with zf:
        infos = zf.infolist()
        if not infos or len(infos) > MAX_BUNDLE_MEMBERS:
            raise BundleError("invalid_bundle")
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise BundleError("invalid_bundle")
        for info in infos:
            _check_member(info)
        if sum(info.file_size for info in infos) > MAX_BUNDLE_UNCOMPRESSED:
            raise BundleError("invalid_bundle")
        if set(names) != set(BUNDLE_MEMBERS):
            raise BundleError("invalid_bundle")
        members = {}
        remaining = MAX_BUNDLE_UNCOMPRESSED
        for info in infos:
            data = _read_member(zf, info, remaining)
            remaining -= len(data)
            members[info.filename] = data
    return members


# -- bundle-level checks ------------------------------------------------------


def _parse_bundle_doc(data: bytes) -> dict:
    try:
        doc = json.loads(data)
    except (UnicodeDecodeError, ValueError):
        raise BundleError("invalid_bundle") from None
    if not isinstance(doc, dict) or set(doc) != {"bundle_version", "files"}:
        raise BundleError("invalid_bundle")
    version = doc["bundle_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise BundleError("invalid_bundle")
    if version != BUNDLE_VERSION:
        raise BundleError("unsupported_bundle_version")
    files = doc["files"]
    if not isinstance(files, dict) or set(files) != set(_DIGESTED_MEMBERS):
        raise BundleError("invalid_bundle")
    if not all(_is_hex64(expected) for expected in files.values()):
        raise BundleError("invalid_bundle")
    return doc


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _check_digests(doc: dict, members: dict) -> None:
    for name in _DIGESTED_MEMBERS:
        if doc["files"][name] != digest_bytes(members[name]):
            raise BundleError("invalid_bundle")


# -- material and trust checks ------------------------------------------------


def _bind_materials(members: dict) -> dict:
    """Structural and digest binding checks over the bundle materials.

    Mirrors proof_materials.bind_materials but leaves the model and
    quantization identity to the trusted-mapping step (so a conflict there
    reports untrusted_model rather than invalid_proof_material), and
    additionally binds the standalone instances.json member.
    """
    try:
        manifest = json.loads(members["manifest.json"])
    except (UnicodeDecodeError, ValueError):
        raise BundleError("invalid_proof_material") from None
    try:
        validate_manifest_structure(manifest)
        settings_bytes = members["settings.json"]
        parse_settings(settings_bytes)
        proof_bytes = members["proof.json"]
        proof_doc = parse_proof(proof_bytes)
        instances = proof_doc["instances"]
        output_felt = decode_output_felt(instances)
        try:
            instances_doc = json.loads(members["instances.json"])
        except (UnicodeDecodeError, ValueError):
            raise MaterialError from None

        materials = manifest["materials"]
        if materials["proof_sha256"] != digest_bytes(proof_bytes):
            raise MaterialError
        if materials["settings_sha256"] != digest_bytes(settings_bytes):
            raise MaterialError
        if materials["verification_key_sha256"] != digest_bytes(
                members["verification_key.key"]):
            raise MaterialError
        # The standalone instances member and the instances embedded in the
        # proof must both be exactly what the manifest pins.
        if manifest["instances_sha256"] != digest_json(instances):
            raise MaterialError
        if digest_json(instances_doc) != manifest["instances_sha256"]:
            raise MaterialError
        if output_felt != manifest["public_output"]["felt"]:
            raise MaterialError
        if abs(output_felt - manifest["q_score"]) > Q_SCORE_TOLERANCE:
            raise MaterialError
        if statement_digest_from_manifest(manifest) != manifest["statement_sha256"]:
            raise MaterialError
    except MaterialError:
        raise BundleError("invalid_proof_material") from None
    return {"manifest": manifest, "instances": instances}


def _check_trusted_mapping(manifest: dict, settings_bytes: bytes,
                           trusted: dict) -> None:
    """The claim must be for the audited release model and quantization."""
    try:
        settings_match = (
            digest_bytes(settings_bytes) == trusted["settings_sha256"]
            or settings_without_timestamp(settings_bytes)
            == settings_without_timestamp(trusted["settings_bytes"])
        )
    except MaterialError:
        settings_match = False
    if manifest["model_id"] != MODEL_ID \
            or manifest["quantization_id"] != QUANTIZATION_ID \
            or manifest["scale"] != SCALE \
            or manifest["rounding"] != ROUNDING \
            or manifest["model_sha256"] != trusted["model_sha256"] \
            or manifest["materials"]["verification_key_sha256"] != trusted["vk_sha256"] \
            or not settings_match:
        raise BundleError("untrusted_model")


def verify_bundle(bundle: Path, backend_dir: Path | None = None) -> dict:
    """Verify an evidence bundle standalone and return the claim on success.

    Never starts the HTTP service and never opens the runtime job store.
    Raises BundleError with one of: invalid_bundle,
    unsupported_bundle_version, invalid_proof_material, untrusted_model;
    ProofBackendError("proof_backend_unavailable") when EZKL is missing.
    """
    members = _read_archive(bundle)
    doc = _parse_bundle_doc(members["bundle.json"])
    _check_digests(doc, members)

    # Everything below needs the real backend (trusted references and the
    # final EZKL verify), so its absence wins over material findings.
    backend = EzklBackend(backend_dir or (ROOT / "runtime" / "backend"))

    bound = _bind_materials(members)
    manifest = bound["manifest"]
    _check_trusted_mapping(manifest, members["settings.json"], backend.trusted())

    try:
        verified = backend.verify_material(
            members["proof.json"], members["settings.json"],
            members["verification_key.key"])
    except ProofBackendError as exc:
        # A failed EZKL verify is invalid material, not a missing backend.
        if exc.code == "invalid_proof_material":
            raise BundleError("invalid_proof_material") from None
        raise
    if not verified:
        raise BundleError("invalid_proof_material")
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
