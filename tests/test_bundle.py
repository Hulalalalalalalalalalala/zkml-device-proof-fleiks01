"""Tests for versioned evidence bundle export and standalone verification.

Every bundle exercised here is built from a genuine EZKL 23.0.5 proof (see
conftest.proof_backend); no mocking or placeholder artifacts are used.
"""

import json
import zipfile

import pytest

from device_proof import __main__ as cli
from device_proof.bundle import (
    BUNDLE_MEMBERS,
    BUNDLE_VERSION,
    BundleError,
    export_bundle,
    verify_bundle,
)
from device_proof.proof_materials import (
    b64_decode,
    digest_bytes,
    statement_digest_from_manifest,
)

from conftest import BACKEND_DIR, SAMPLE, wait_for


@pytest.fixture()
def succeeded_job(manager, tmp_path):
    """One real succeeded proof job plus its runtime root and materials."""
    from device_proof.scoring import ScoreRequest

    record = manager.submit(ScoreRequest.model_validate(SAMPLE))
    done = wait_for(manager, record["job_id"])
    assert done["status"] == "succeeded"
    materials = manager.store.load_material_bytes(record["job_id"])
    return {
        "job_id": record["job_id"],
        "runtime_root": manager.root,
        "materials": materials,
    }


def _export(succeeded_job, output):
    export_bundle(
        succeeded_job["job_id"], output, succeeded_job["runtime_root"])
    return output


def _read_members(bundle_path):
    with zipfile.ZipFile(bundle_path) as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def _repack(members, path):
    """Write members back as a plain zip (test helper for tampering)."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _repack_with_digests(members, path):
    """Repack after repairing bundle.json digests over the other members."""
    bundle_doc = json.loads(members["bundle.json"])
    for name in bundle_doc["files"]:
        bundle_doc["files"][name] = digest_bytes(members[name])
    members["bundle.json"] = json.dumps(bundle_doc, sort_keys=True).encode()
    return _repack(members, path)


# -- export -------------------------------------------------------------------


def test_export_succeeded_job_layout_and_digests(succeeded_job, tmp_path):
    bundle = _export(succeeded_job, tmp_path / "evidence.zip")
    with zipfile.ZipFile(bundle) as zf:
        # Root contains exactly the six fixed members, nothing else.
        assert sorted(info.filename for info in zf.infolist()) == \
            sorted(BUNDLE_MEMBERS)
        assert all(not info.is_dir() for info in zf.infolist())
        members = {info.filename: zf.read(info) for info in zf.infolist()}
    doc = json.loads(members["bundle.json"])
    assert doc["bundle_version"] == BUNDLE_VERSION
    assert set(doc) == {"bundle_version", "files"}
    assert set(doc["files"]) == set(BUNDLE_MEMBERS) - {"bundle.json"}
    for name, expected in doc["files"].items():
        assert expected == digest_bytes(members[name])
    # The bundle carries exactly the persisted public materials.
    for name, data in succeeded_job["materials"].items():
        assert members[name] == data
    # No features, encodings, witness, or internal paths anywhere.
    text = b"".join(members.values()).decode("utf-8", errors="ignore")
    for forbidden in ("features", "witness", "temperature", "input_data",
                      "runtime/", "materials/"):
        assert forbidden not in text


def test_export_unknown_job(succeeded_job, tmp_path):
    with pytest.raises(BundleError) as excinfo:
        export_bundle("f" * 32, tmp_path / "x.zip",
                      succeeded_job["runtime_root"])
    assert excinfo.value.code == "proof_job_not_found"


def test_export_not_ready_job(manager, tmp_path):
    # A queued (never processed) record: any non-succeeded state is not ready.
    record = manager.store.create({"model_id": "device-health-v1"})
    with pytest.raises(BundleError) as excinfo:
        export_bundle(record["job_id"], tmp_path / "x.zip", manager.root)
    assert excinfo.value.code == "proof_bundle_not_ready"


def test_export_cli_codes(succeeded_job, tmp_path, capsys):
    out = tmp_path / "cli.zip"
    cli.main(["export-bundle", "--job-id", succeeded_job["job_id"],
              "--output", str(out),
              "--runtime-dir", str(succeeded_job["runtime_root"])])
    assert out.exists()
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["export-bundle", "--job-id", "0" * 32,
                  "--output", str(tmp_path / "y.zip"),
                  "--runtime-dir", str(succeeded_job["runtime_root"])])
    assert excinfo.value.code == 1
    assert capsys.readouterr().err.strip() == "proof_job_not_found"


# -- standalone verification --------------------------------------------------


def test_verify_bundle_success(succeeded_job, tmp_path):
    bundle = _export(succeeded_job, tmp_path / "evidence.zip")
    result = verify_bundle(bundle, backend_dir=BACKEND_DIR)
    # Same response shape as POST /proof-verifications.
    assert result == {
        "verified": True,
        "model_id": "device-health-v1",
        "model_sha256": result["model_sha256"],
        "quantization_id": "device-health-v1-q16-v1",
        "q_score": 26214,
        "statement_sha256": result["statement_sha256"],
        "instances": result["instances"],
        "ezkl_version": "23.0.5",
    }
    assert len(result["model_sha256"]) == 64
    assert len(result["statement_sha256"]) == 64
    assert len(result["instances"]) == 1
    assert len(result["instances"][0]) == 1


def test_verify_bundle_cli_success(succeeded_job, tmp_path, capsys):
    bundle = _export(succeeded_job, tmp_path / "evidence.zip")
    cli.main(["verify-bundle", "--bundle", str(bundle),
              "--backend-dir", str(BACKEND_DIR)])
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["verified"] is True
    assert body["q_score"] == 26214


def test_verify_bundle_rejects_non_zip(tmp_path):
    path = tmp_path / "not.zip"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(path, backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_bundle"


@pytest.mark.parametrize("name", [
    "/abs/bundle.json",
    "../bundle.json",
    "sub/../bundle.json",
    "C:/bundle.json",
    "dir\\bundle.json",
])
def test_verify_bundle_rejects_bad_member_names(succeeded_job, tmp_path, name):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    members[name] = members.pop("bundle.json")
    evil = _repack(members, tmp_path / "evil.zip")
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(evil, backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_bundle"


def test_verify_bundle_rejects_duplicate_members(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    path = tmp_path / "dup.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for member, data in members.items():
            zf.writestr(member, data)
        zf.writestr("proof.json", members["proof.json"])
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(path, backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_bundle"


def test_verify_bundle_rejects_directories_and_links(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    for extra in ("dir/", "link"):
        path = tmp_path / "evil.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for member, data in members.items():
                zf.writestr(member, data)
            if extra.endswith("/"):
                zf.writestr(extra, b"")
            else:
                info = zipfile.ZipInfo(extra)
                info.external_attr = 0o120777 << 16  # symlink
                zf.writestr(info, b"target")
        with pytest.raises(BundleError) as excinfo:
            verify_bundle(path, backend_dir=BACKEND_DIR)
        assert excinfo.value.code == "invalid_bundle"


def test_verify_bundle_rejects_member_flood(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    for index in range(3):
        members[f"extra{index}.bin"] = b"x"
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack(members, tmp_path / "flood.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_bundle"


def test_verify_bundle_rejects_zip_bomb(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    # Highly compressible huge member: tiny archive, enormous expansion.
    members["proof.json"] = b"0" * (128 * 1024 * 1024)
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack(members, tmp_path / "bomb.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_bundle"


def test_verify_bundle_rejects_unknown_version(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    doc = json.loads(members["bundle.json"])
    doc["bundle_version"] = 2
    members["bundle.json"] = json.dumps(doc, sort_keys=True).encode()
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack(members, tmp_path / "v2.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "unsupported_bundle_version"


def test_verify_bundle_rejects_digest_mismatch(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    # Tamper a member without repairing the bundle.json digest map.
    members["instances.json"] = b"[]"
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack(members, tmp_path / "tampered.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_bundle"


def test_verify_bundle_rejects_material_tampering(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    manifest = json.loads(members["manifest.json"])
    manifest["q_score"] = 999
    members["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack_with_digests(members, tmp_path / "bad.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_proof_material"


def test_verify_bundle_rejects_instances_mismatch(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    # instances.json no longer matches what the manifest pins.
    members["instances.json"] = json.dumps([["0" * 64]]).encode()
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack_with_digests(members, tmp_path / "bad.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_proof_material"


def test_verify_bundle_rejects_untrusted_model(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    manifest = json.loads(members["manifest.json"])
    manifest["model_sha256"] = "0" * 64
    manifest["statement_sha256"] = statement_digest_from_manifest(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack_with_digests(members, tmp_path / "bad.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "untrusted_model"


def test_verify_bundle_rejects_relabeled_quantization(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    manifest = json.loads(members["manifest.json"])
    manifest["quantization_id"] = "other-quantization-v9"
    manifest["statement_sha256"] = statement_digest_from_manifest(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack_with_digests(members, tmp_path / "bad.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "untrusted_model"


def test_verify_bundle_rejects_forged_proof(succeeded_job, tmp_path):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    # Mutate proof bytes, then repair every offline digest so only the real
    # EZKL verifier can reject it.
    proof_doc = json.loads(members["proof.json"])
    proof_doc["proof"][0] ^= 0x01
    forged = json.dumps(proof_doc, separators=(",", ":")).encode()
    members["proof.json"] = forged
    manifest = json.loads(members["manifest.json"])
    manifest["materials"]["proof_sha256"] = digest_bytes(forged)
    members["manifest.json"] = json.dumps(manifest).encode()
    with pytest.raises(BundleError) as excinfo:
        verify_bundle(_repack_with_digests(members, tmp_path / "forged.zip"),
                      backend_dir=BACKEND_DIR)
    assert excinfo.value.code == "invalid_proof_material"


def test_verify_bundle_cli_failure_codes(succeeded_job, tmp_path, capsys):
    members = _read_members(_export(succeeded_job, tmp_path / "ok.zip"))
    doc = json.loads(members["bundle.json"])
    doc["bundle_version"] = 99
    members["bundle.json"] = json.dumps(doc, sort_keys=True).encode()
    bad = _repack(members, tmp_path / "bad.zip")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["verify-bundle", "--bundle", str(bad),
                  "--backend-dir", str(BACKEND_DIR)])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "unsupported_bundle_version"
    assert captured.out == ""


def test_verify_bundle_cli_backend_unavailable(succeeded_job, tmp_path,
                                               capsys, monkeypatch):
    from device_proof import bundle as bundle_mod
    from device_proof.backend import ProofBackendError

    class BrokenBackend:
        def __init__(self, *args, **kwargs):
            raise ProofBackendError("proof_backend_unavailable")

    monkeypatch.setattr(bundle_mod, "EzklBackend", BrokenBackend)
    bundle = _export(succeeded_job, tmp_path / "evidence.zip")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["verify-bundle", "--bundle", str(bundle)])
    assert excinfo.value.code == 1
    assert capsys.readouterr().err.strip() == "proof_backend_unavailable"


def test_b64_materials_match_bundle(succeeded_job, tmp_path, manager):
    # The bundle carries the same bytes the HTTP view exposes as Base64.
    bundle = _export(succeeded_job, tmp_path / "evidence.zip")
    members = _read_members(bundle)
    record = manager.get(succeeded_job["job_id"])
    view = record["materials"]
    assert members["proof.json"] == b64_decode(view["proof"])
    assert members["verification_key.key"] == b64_decode(view["verification_key"])
    assert members["settings.json"] == b64_decode(view["settings"])
    assert json.loads(members["manifest.json"]) == view["manifest"]
    assert json.loads(members["instances.json"]) == view["instances"]
