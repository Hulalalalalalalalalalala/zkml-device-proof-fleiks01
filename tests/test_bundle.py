"""End-to-end tests for versioned evidence bundles (export-bundle/verify-bundle).

Bundles under test are exported from genuine EZKL 23.0.5 proof jobs (see
conftest.proof_backend); verification runs the real EZKL verifier. No
mocking or placeholder artifacts are used.
"""

import json
import zipfile

import pytest

from device_proof.__main__ import main
from device_proof.backend import ProofBackendError
from device_proof.proof_materials import digest_bytes, statement_digest_from_manifest

from conftest import BACKEND_DIR, SAMPLE, wait_for


def _succeeded_job(client, timeout=40.0):
    import time

    job_id = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/proof-jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            assert body["status"] == "succeeded"
            return job_id
        time.sleep(0.05)
    raise AssertionError("job never finished")


def _read_zip(path):
    with zipfile.ZipFile(path) as zf:
        return {info.filename: zf.read(info.filename) for info in zf.infolist()}


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)


def _export(job_id, output, runtime_root, capsys):
    main(["export-bundle", "--job-id", job_id, "--output", str(output),
          "--runtime-dir", str(runtime_root)])
    capsys.readouterr()  # drain the success line
    return output


def _verify_cli(bundle, capsys):
    main(["verify-bundle", "--bundle", str(bundle),
          "--backend-dir", str(BACKEND_DIR)])
    out, err = capsys.readouterr()
    assert err == ""
    return json.loads(out)


def _verify_cli_fails(bundle, capsys, code):
    with pytest.raises(SystemExit):
        main(["verify-bundle", "--bundle", str(bundle),
              "--backend-dir", str(BACKEND_DIR)])
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == code


@pytest.fixture()
def bundle(manager, client, tmp_path, capsys):
    """One real exported bundle from a genuine succeeded proof job."""
    job_id = _succeeded_job(client)
    return _export(job_id, tmp_path / "bundle.zip", manager.root, capsys)


# -- export -------------------------------------------------------------------

def test_export_bundle_succeeded_job(bundle):
    members = _read_zip(bundle)
    assert set(members) == {
        "bundle.json", "manifest.json", "proof.json",
        "verification_key.key", "settings.json", "instances.json",
    }
    doc = json.loads(members["bundle.json"])
    assert doc["bundle_version"] == 1
    assert set(doc["members"]) == {
        "manifest.json", "proof.json", "verification_key.key",
        "settings.json", "instances.json",
    }
    for name, digest in doc["members"].items():
        assert digest_bytes(members[name]) == digest
    manifest = json.loads(members["manifest.json"])
    assert manifest["model_id"] == "device-health-v1"
    assert manifest["q_score"] == 26214
    # bundle.json carries only the version and member digests: no features,
    # encodings, witnesses, or internal paths.
    bundle_text = members["bundle.json"].decode()
    assert set(doc) == {"bundle_version", "members"}
    for token in ("temperature", "vibration", "current", "runtime",
                  "witness", "features", "input_data", "q_t", "/"):
        assert token not in bundle_text


def test_export_bundle_cli_success_json(manager, client, tmp_path, capsys):
    job_id = _succeeded_job(client)
    output = tmp_path / "out.zip"
    main(["export-bundle", "--job-id", job_id, "--output", str(output),
          "--runtime-dir", str(manager.root)])
    out, err = capsys.readouterr()
    assert err == ""
    body = json.loads(out)
    assert body == {"job_id": job_id, "bundle_version": 1, "output": str(output)}


def test_export_bundle_unknown_job(manager, tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["export-bundle", "--job-id", "f" * 32,
              "--output", str(tmp_path / "x.zip"),
              "--runtime-dir", str(manager.root)])
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "proof_job_not_found"


def test_export_bundle_failed_job_not_ready(manager, client, tmp_path, capsys):
    def boom(features, job_dir):
        raise RuntimeError("internal detail that must not surface")

    manager.backend().prove = boom
    job_id = client.post("/proof-jobs", json=SAMPLE).json()["job_id"]
    assert wait_for(manager, job_id)["status"] == "failed"
    with pytest.raises(SystemExit):
        main(["export-bundle", "--job-id", job_id,
              "--output", str(tmp_path / "x.zip"),
              "--runtime-dir", str(manager.root)])
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "proof_bundle_not_ready"


# -- verify: happy path --------------------------------------------------------

def test_verify_bundle_valid(bundle, capsys):
    body = _verify_cli(bundle, capsys)
    assert body["verified"] is True
    assert body["model_id"] == "device-health-v1"
    assert body["quantization_id"] == "device-health-v1-q16-v1"
    assert body["q_score"] == 26214
    assert len(body["statement_sha256"]) == 64
    assert len(body["instances"]) == 1 and len(body["instances"][0]) == 1
    assert body["ezkl_version"] == "23.0.5"


# -- verify: version / structure / safety -------------------------------------

def test_verify_bundle_unsupported_version(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    doc = json.loads(members["bundle.json"])
    doc["bundle_version"] = 2
    members["bundle.json"] = json.dumps(doc).encode()
    _write_zip(tmp_path / "v2.zip", members)
    _verify_cli_fails(tmp_path / "v2.zip", capsys, "unsupported_bundle_version")


def test_verify_bundle_rejects_missing_and_extra_members(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    del members["proof.json"]
    _write_zip(tmp_path / "missing.zip", members)
    _verify_cli_fails(tmp_path / "missing.zip", capsys, "invalid_bundle")

    members = _read_zip(bundle)
    members["extra.txt"] = b"x"
    _write_zip(tmp_path / "extra.zip", members)
    _verify_cli_fails(tmp_path / "extra.zip", capsys, "invalid_bundle")


def test_verify_bundle_rejects_unsafe_members(bundle, tmp_path, capsys):
    good = _read_zip(bundle)
    for evil in ("../evil", "/abs", "C:\\win", "dir/", "proof.json/../x"):
        members = dict(good)
        members[evil] = b"x"
        _write_zip(tmp_path / "evil.zip", members)
        _verify_cli_fails(tmp_path / "evil.zip", capsys, "invalid_bundle")


def test_verify_bundle_rejects_duplicate_members(bundle, tmp_path, capsys):
    with zipfile.ZipFile(bundle) as zf:
        infos = [(i, zf.read(i.filename)) for i in zf.infolist()]
    with zipfile.ZipFile(tmp_path / "dup.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for info, blob in infos:
            zf.writestr(info.filename, blob)
        zf.writestr("proof.json", b"duplicate")
    _verify_cli_fails(tmp_path / "dup.zip", capsys, "invalid_bundle")


def test_verify_bundle_rejects_symlink(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    path = tmp_path / "link.zip"
    _write_zip(path, members)
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16  # Unix symlink
        zf.writestr(info, b"/etc/passwd")
    _verify_cli_fails(path, capsys, "invalid_bundle")


def test_verify_bundle_rejects_oversized_member(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    members["proof.json"] = b"0" * (33 * 1024 * 1024)
    _write_zip(tmp_path / "big.zip", members)
    _verify_cli_fails(tmp_path / "big.zip", capsys, "invalid_bundle")


def test_verify_bundle_rejects_non_zip(tmp_path, capsys):
    path = tmp_path / "not.zip"
    path.write_bytes(b"this is not a zip archive")
    _verify_cli_fails(path, capsys, "invalid_bundle")


# -- verify: material / model / backend ---------------------------------------

def test_verify_bundle_rejects_digest_tampering(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    manifest = json.loads(members["manifest.json"])
    manifest["q_score"] = 999
    members["manifest.json"] = json.dumps(manifest).encode()
    _write_zip(tmp_path / "tampered.zip", members)
    _verify_cli_fails(tmp_path / "tampered.zip", capsys, "invalid_proof_material")


def test_verify_bundle_rejects_instances_mismatch(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    # A different (but well-formed) instances file, with bundle.json repaired
    # so only the manifest/proof ↔ instances binding can catch it.
    instances = json.loads(members["instances.json"])
    instances[0][0] = "00" * 32
    members["instances.json"] = json.dumps(instances).encode()
    doc = json.loads(members["bundle.json"])
    doc["members"]["instances.json"] = digest_bytes(members["instances.json"])
    members["bundle.json"] = json.dumps(doc).encode()
    _write_zip(tmp_path / "inst.zip", members)
    _verify_cli_fails(tmp_path / "inst.zip", capsys, "invalid_proof_material")


def _forged_proof_bundle(bundle, tmp_path):
    members = _read_zip(bundle)
    proof_doc = json.loads(members["proof.json"])
    proof_doc["proof"][0] ^= 0x01
    forged = json.dumps(proof_doc, separators=(",", ":")).encode()
    members["proof.json"] = forged
    # Repair the manifest proof digest and the bundle.json digests so every
    # offline check passes and EZKL itself must reject the forged proof.
    manifest = json.loads(members["manifest.json"])
    manifest["materials"]["proof_sha256"] = digest_bytes(forged)
    members["manifest.json"] = json.dumps(manifest).encode()
    doc = json.loads(members["bundle.json"])
    doc["members"]["proof.json"] = digest_bytes(forged)
    doc["members"]["manifest.json"] = digest_bytes(members["manifest.json"])
    members["bundle.json"] = json.dumps(doc).encode()
    _write_zip(tmp_path / "forged.zip", members)
    return tmp_path / "forged.zip"


def test_verify_bundle_rejects_forged_proof(bundle, tmp_path, capsys):
    _verify_cli_fails(_forged_proof_bundle(bundle, tmp_path), capsys,
                      "invalid_proof_material")


def test_verify_bundle_rejects_untrusted_model(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    manifest = json.loads(members["manifest.json"])
    manifest["model_sha256"] = "0" * 64
    manifest["statement_sha256"] = statement_digest_from_manifest(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    doc = json.loads(members["bundle.json"])
    doc["members"]["manifest.json"] = digest_bytes(members["manifest.json"])
    members["bundle.json"] = json.dumps(doc).encode()
    _write_zip(tmp_path / "model.zip", members)
    _verify_cli_fails(tmp_path / "model.zip", capsys, "untrusted_model")


def test_verify_bundle_rejects_relabeled_quantization(bundle, tmp_path, capsys):
    members = _read_zip(bundle)
    manifest = json.loads(members["manifest.json"])
    manifest["quantization_id"] = "other-quantization-v9"
    manifest["statement_sha256"] = statement_digest_from_manifest(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    doc = json.loads(members["bundle.json"])
    doc["members"]["manifest.json"] = digest_bytes(members["manifest.json"])
    members["bundle.json"] = json.dumps(doc).encode()
    _write_zip(tmp_path / "quant.zip", members)
    _verify_cli_fails(tmp_path / "quant.zip", capsys, "untrusted_model")


def test_verify_bundle_backend_unavailable(bundle, monkeypatch, capsys):
    class BrokenBackend:
        def __init__(self, *args, **kwargs):
            raise ProofBackendError("proof_backend_unavailable")

    monkeypatch.setattr("device_proof.bundle.EzklBackend", BrokenBackend)
    _verify_cli_fails(bundle, capsys, "proof_backend_unavailable")
