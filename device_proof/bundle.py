"""Versioned evidence bundles: export and offline verification.

``export_bundle`` packages the five public materials of one *succeeded*
proof job into a single ZIP whose root contains exactly::

    bundle.json  manifest.json  proof.json
    verification_key.key  settings.json  instances.json

``bundle.json`` carries ``bundle_version = 1`` and the SHA-256 of every
other member; it never contains features, per-feature encodings,
witnesses, or internal filesystem paths.

``verify_bundle`` re-checks a bundle without starting HTTP or opening the
runtime job store: it enforces ZIP safety limits (member count, total
expanded size; no absolute paths, ``..``, duplicates, directories, links,
or compression bombs), then the bundle version and member set, then every
digest and the manifest/instances binding, then the trusted model and
quantization mapping, and finally the real EZKL verifier. Failures raise
``BundleError`` with one stable code:

* ``unsupported_bundle_version`` – unknown ``bundle_version``;
* ``invalid_bundle``             – bundle structure or safety violation;
* ``invalid_proof_material``     – digest/binding/material/verify failure;
* ``untrusted_model``            – model or quantization mapping conflict;
* ``proof_backend_unavailable``  – no usable local EZKL backend.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import zipfile

from .audit import MODEL_ID
from .backend import EzklBackend, ProofBackendError
from .jobs import SUCCEEDED, JobStore
from .proof_materials import (
    MaterialError,
    bind_materials,
    digest_bytes,
    settings_without_timestamp,
)
from .scoring import ROOT
from .statements import QUANTIZATION_ID, ROUNDING, SCALE

BUNDLE_VERSION = 1

# Root members, in archive order. bundle.json binds the other five.
BUNDLE_MEMBERS = (
    "bundle.json",
    "manifest.json",
    "proof.json",
    "verification_key.key",
    "settings.json",
    "instances.json",
)
_MATERIAL_MEMBERS = BUNDLE_MEMBERS[1:]

# Safety limits for untrusted archives: far above any legitimate bundle
# (a real bundle is well under 1 MiB) and small enough to defuse bombs.
_MAX_MEMBERS = 16
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024

_HEX_CHARS = set("0123456789abcdef")


class BundleError(Exception):
    """Bundle export/verification failure carrying a stable, public code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _is_hex64(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_CHARS for ch in value)
    )


# -- export -------------------------------------------------------------------


def export_bundle(job_id: str, output_path: Path, *,
                  runtime_root: Path | None = None) -> dict:
    """Write the versioned evidence bundle for one succeeded job.

    Raises BundleError("proof_job_not_found") for an unknown job and
    BundleError("proof_bundle_not_ready") for any non-succeeded status or
    incomplete materials. The bundle contains public materials only.
    """
    store = JobStore(runtime_root or (ROOT / "runtime"))
    record = store.get(job_id)
    if record is None:
        raise BundleError("proof_job_not_found")
    if record["status"] != SUCCEEDED:
        raise BundleError("proof_bundle_not_ready")
    materials = store.load_materials(job_id)
    if materials is None:
        raise BundleError("proof_bundle_not_ready")

    files = {
        "manifest.json": json.dumps(
            materials["manifest"], ensure_ascii=False, sort_keys=True
        ).encode("utf-8"),
        "proof.json": materials["proof"],
        "verification_key.key": materials["verification_key"],
        "settings.json": materials["settings"],
        "instances.json": json.dumps(
            materials["instances"], ensure_ascii=False
        ).encode("utf-8"),
    }
    bundle_doc = {
        "bundle_version": BUNDLE_VERSION,
        "members": {
            name: sha256(files[name]).hexdigest() for name in _MATERIAL_MEMBERS
        },
    }
    files["bundle.json"] = json.dumps(
        bundle_doc, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")

    output_path = Path(output_path)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        # Fixed timestamps keep the archive byte-reproducible; permissions
        # are pinned to a plain regular file (never a link or directory).
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in BUNDLE_MEMBERS:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100600 << 16
                zf.writestr(info, files[name])
        os.replace(tmp_path, output_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise BundleError("output_unwritable") from None
    return bundle_doc


# -- verification -------------------------------------------------------------


def _reject_link_or_special(info) -> bool:
    # Unix mode bits (when present) must describe a regular file; this
    # rejects symlinks, fifos, devices, and anything else non-regular.
    mode = (info.external_attr >> 16) & 0o170000
    return mode not in (0, 0o100000)


def _read_bundle_zip(bundle_path: Path) -> dict:
    """Open an untrusted bundle and return ``{name: bytes}`` for its members.

    Every structural or safety violation raises BundleError("invalid_bundle").
    """
    try:
        zf = zipfile.ZipFile(bundle_path)
    except (OSError, zipfile.BadZipFile):
        raise BundleError("invalid_bundle") from None
    with zf:
        infos = zf.infolist()
        if not infos or len(infos) > _MAX_MEMBERS:
            raise BundleError("invalid_bundle")
        names: set[str] = set()
        declared_total = 0
        for info in infos:
            name = info.filename
            if info.is_dir() or name.endswith("/"):
                raise BundleError("invalid_bundle")
            if _reject_link_or_special(info):
                raise BundleError("invalid_bundle")
            if info.flag_bits & 0x1:  # encrypted member
                raise BundleError("invalid_bundle")
            if (
                name.startswith(("/", "\\"))
                or "\\" in name
                or (len(name) >= 2 and name[1] == ":")
            ):
                raise BundleError("invalid_bundle")
            if any(part in ("", ".", "..") for part in name.split("/")):
                raise BundleError("invalid_bundle")
            if name in names:
                raise BundleError("invalid_bundle")
            names.add(name)
            declared_total += info.file_size
            if info.file_size > _MAX_MEMBER_BYTES or declared_total > _MAX_TOTAL_BYTES:
                raise BundleError("invalid_bundle")
        if names != set(BUNDLE_MEMBERS):
            raise BundleError("invalid_bundle")
        members = {}
        total = 0
        for info in infos:
            # Read with a hard cap: declared sizes cannot be trusted.
            try:
                with zf.open(info) as handle:
                    data = handle.read(_MAX_MEMBER_BYTES + 1)
            except (NotImplementedError, RuntimeError, zipfile.BadZipFile):
                raise BundleError("invalid_bundle") from None
            if len(data) > _MAX_MEMBER_BYTES:
                raise BundleError("invalid_bundle")
            total += len(data)
            if total > _MAX_TOTAL_BYTES:
                raise BundleError("invalid_bundle")
            members[info.filename] = data
    return members


def _parse_bundle_doc(raw: bytes) -> dict:
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise BundleError("invalid_bundle") from None
    if not isinstance(doc, dict) or set(doc) != {"bundle_version", "members"}:
        raise BundleError("invalid_bundle")
    version = doc["bundle_version"]
    if isinstance(version, bool) or version != BUNDLE_VERSION:
        raise BundleError("unsupported_bundle_version")
    members = doc["members"]
    if not isinstance(members, dict) or set(members) != set(_MATERIAL_MEMBERS):
        raise BundleError("invalid_bundle")
    if not all(_is_hex64(digest) for digest in members.values()):
        raise BundleError("invalid_bundle")
    return doc


def verify_bundle(bundle_path: Path, *, backend_dir: Path | None = None,
                  backend: EzklBackend | None = None) -> dict:
    """Verify a bundle end to end; return the standard verification response.

    Never starts HTTP and never opens the runtime job store. Raises
    BundleError (or ProofBackendError when the local EZKL backend is
    unavailable) with a stable code on any failure.
    """
    members = _read_bundle_zip(Path(bundle_path))
    bundle_doc = _parse_bundle_doc(members["bundle.json"])

    # Member digests bind every material file to bundle.json.
    for name, expected in bundle_doc["members"].items():
        if sha256(members[name]).hexdigest() != expected:
            raise BundleError("invalid_proof_material")

    try:
        manifest = json.loads(members["manifest.json"])
        instances = json.loads(members["instances.json"])
    except (UnicodeDecodeError, ValueError):
        raise BundleError("invalid_proof_material") from None
    proof_bytes = members["proof.json"]
    settings_bytes = members["settings.json"]
    vk_bytes = members["verification_key.key"]

    # Structure and digest/binding checks (manifest ↔ proof/settings/vk and
    # manifest ↔ instances), without pinning the model identity yet so a
    # conflict there is reported as untrusted_model below.
    try:
        bound = bind_materials(
            manifest=manifest, proof_bytes=proof_bytes,
            settings_bytes=settings_bytes, vk_bytes=vk_bytes, pin_model=False)
    except MaterialError:
        raise BundleError("invalid_proof_material") from None
    if bound["instances"] != instances:
        raise BundleError("invalid_proof_material")

    # Trusted model and quantization mapping: the claim must be for the
    # audited release model under the pinned quantization scheme, and the
    # verification key/settings must be the trusted circuit's.
    if backend is None:
        backend = EzklBackend(backend_dir or (ROOT / "runtime" / "backend"))
    trusted = backend.trusted()
    settings_match = (
        digest_bytes(settings_bytes) == trusted["settings_sha256"]
        or settings_without_timestamp(settings_bytes)
        == settings_without_timestamp(trusted["settings_bytes"])
    )
    if (
        manifest["model_id"] != MODEL_ID
        or manifest["model_sha256"] != trusted["model_sha256"]
        or manifest["materials"]["verification_key_sha256"] != trusted["vk_sha256"]
        or manifest["quantization_id"] != QUANTIZATION_ID
        or manifest["scale"] != SCALE
        or manifest["rounding"] != ROUNDING
        or not settings_match
    ):
        raise BundleError("untrusted_model")

    try:
        verified = backend.verify_material(proof_bytes, settings_bytes, vk_bytes)
    except ProofBackendError as exc:
        # A failed EZKL verify means invalid material, not a missing backend.
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
