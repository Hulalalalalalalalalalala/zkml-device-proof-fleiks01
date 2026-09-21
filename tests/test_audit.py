import copy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json

from fastapi.testclient import TestClient
import pytest

from device_proof import __main__, audit
from device_proof.api import app
from device_proof.audit import MODEL_ID, RELEASE_SHA256, AuditError, audit_model
from device_proof.scoring import ROOT


client = TestClient(app)
SAMPLE = json.loads((ROOT / "examples" / "sample.json").read_text())
MANIFEST = json.loads((ROOT / "models" / "device-health-v1.json").read_text())
MODEL_BYTES = (ROOT / "models" / "device-health-v1.onnx").read_bytes()


def test_audit_report():
    report = audit_model()
    assert report["model_id"] == MODEL_ID
    assert report["status"] == "ok"
    assert report["mode"] == "ordinary-inference"
    assert report["sha256"] == RELEASE_SHA256
    assert report["onnx_opset"] == 13
    assert report["onnx_ir_version"] == 8
    assert report["reference_scores"]["zeros"] == pytest.approx(0.1, abs=1e-6)
    assert report["reference_scores"]["sample"] == pytest.approx(0.4, abs=1e-6)
    assert report["reference_scores"]["ones"] == pytest.approx(1.0, abs=1e-6)


def test_integrity_endpoint_matches_audit():
    response = client.get(f"/models/{MODEL_ID}/integrity")
    assert response.status_code == 200
    assert response.json() == audit_model()


def test_integrity_unknown_model():
    assert client.get("/models/absent/integrity").status_code == 404


def test_model_check_cli_ok(capsys):
    __main__.main(["model-check"])
    out, err = capsys.readouterr()
    assert err == ""
    report = json.loads(out)
    assert report["model_id"] == MODEL_ID
    assert report["status"] == "ok"
    assert len(out.strip().splitlines()) == 1


def test_model_check_cli_unknown_model(capsys):
    with pytest.raises(SystemExit) as excinfo:
        __main__.main(["model-check", "--model-id", "absent"])
    assert excinfo.value.code != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "unknown_model"


def _break_manifest(tmp_path, monkeypatch, mutate):
    manifest = copy.deepcopy(MANIFEST)
    mutate(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(audit, "MANIFEST_PATH", path)


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda m: m["feature_order"].__setitem__(0, m["feature_order"][1]), "manifest_duplicate_feature"),
        (lambda m: m["feature_order"].__setitem__(0, "humidity"), "manifest_unknown_feature"),
        (lambda m: m.__setitem__("version", "2.0.0"), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("id", "other-model"), "manifest_contract_mismatch"),
        (lambda m: m["feature_order"].reverse(), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("feature_range", [0.0, 2.0]), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("score_range", [0.0, 1.0]), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("input_shape", [1, 5]), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("output_shape", [2, 1]), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("onnx_opset", 12), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("onnx_ir_version", 7), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("sha256", "0" * 64), "manifest_contract_mismatch"),
        (lambda m: m.__setitem__("onnx_opset", "13"), "manifest_invalid_type"),
        (lambda m: m.__setitem__("input_shape", [1, "4"]), "manifest_invalid_type"),
        (lambda m: m.__setitem__("feature_range", [0.0]), "manifest_invalid_type"),
    ],
)
def test_manifest_corruption_rejected(tmp_path, monkeypatch, capsys, mutate, code):
    _break_manifest(tmp_path, monkeypatch, mutate)
    with pytest.raises(AuditError) as excinfo:
        audit_model()
    assert excinfo.value.code == code
    response = client.get(f"/models/{MODEL_ID}/integrity")
    assert response.status_code == 503
    assert response.json()["detail"] == "model_unavailable"
    assert client.post("/score", json=SAMPLE).status_code == 503
    with pytest.raises(SystemExit) as exitinfo:
        __main__.main(["model-check"])
    assert exitinfo.value.code != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == code


def test_manifest_not_json(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(audit, "MANIFEST_PATH", path)
    with pytest.raises(AuditError) as excinfo:
        audit_model()
    assert excinfo.value.code == "manifest_invalid_json"


def test_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "MANIFEST_PATH", tmp_path / "absent.json")
    with pytest.raises(AuditError) as excinfo:
        audit_model()
    assert excinfo.value.code == "manifest_missing"


def test_replaced_model_file_blocked(tmp_path, monkeypatch):
    assert client.post("/score", json=SAMPLE).status_code == 200
    tampered = bytearray(MODEL_BYTES)
    tampered[-1] ^= 0xFF
    path = tmp_path / "model.onnx"
    path.write_bytes(bytes(tampered))
    monkeypatch.setattr(audit, "MODEL_PATH", path)
    with pytest.raises(AuditError) as excinfo:
        audit_model()
    assert excinfo.value.code == "model_digest_mismatch"
    # The previously cached session must not keep serving the replaced file.
    response = client.post("/score", json=SAMPLE)
    assert response.status_code == 503
    assert response.json()["detail"] == "model_unavailable"
    assert client.get("/healthz").status_code == 503


def test_rehashed_model_still_rejected(tmp_path, monkeypatch):
    tampered = bytearray(MODEL_BYTES)
    tampered[-1] ^= 0xFF
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(bytes(tampered))
    monkeypatch.setattr(audit, "MODEL_PATH", model_path)
    _break_manifest(
        tmp_path,
        monkeypatch,
        lambda m: m.__setitem__("sha256", sha256(bytes(tampered)).hexdigest()),
    )
    with pytest.raises(AuditError) as excinfo:
        audit_model()
    assert excinfo.value.code == "manifest_contract_mismatch"


def test_concurrent_audits_return_complete_identical_reports():
    with ThreadPoolExecutor(max_workers=8) as pool:
        reports = list(pool.map(lambda _: audit_model(), range(32)))
    assert all(report == reports[0] for report in reports)


def test_legacy_cli_still_scores(capsys):
    __main__.main([])
    out, err = capsys.readouterr()
    assert err == ""
    assert json.loads(out)["score"] == pytest.approx(0.4, abs=1e-6)
