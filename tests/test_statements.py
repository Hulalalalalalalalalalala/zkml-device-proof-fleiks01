import copy
import hashlib
import json
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from device_proof import __main__
from device_proof.api import app
from device_proof.scoring import ROOT
from device_proof.statements import (
    QUANTIZATION_ID,
    SCALE,
    UINT32_MAX,
    QuantizationError,
    _quantize_feature,
    _round_ties_to_even,
    build_statement,
)
from device_proof.scoring import ScoreRequest, score
from fractions import Fraction


client = TestClient(app)
SAMPLE = json.loads((ROOT / "examples" / "sample.json").read_text())

STATEMENT_KEYS = {
    "model_id",
    "model_sha256",
    "quantization_id",
    "q_score",
    "scale",
    "rounding",
    "statement_sha256",
}


def _statement_body(payload):
    response = client.post("/statements", json=payload)
    assert response.status_code == 200
    return response.json()


def test_sample_statement_fields_and_no_leakage():
    info = client.get("/models/device-health-v1").json()
    body = _statement_body(SAMPLE)
    assert set(body) == STATEMENT_KEYS
    assert body["model_id"] == "device-health-v1"
    assert body["model_sha256"] == info["sha256"]
    assert body["quantization_id"] == QUANTIZATION_ID
    assert body["quantization_id"] == "device-health-v1-q16-v1"
    assert body["scale"] == 65536
    assert body["rounding"] == "ties-to-even"
    assert isinstance(body["q_score"], int)
    assert 0 <= body["q_score"] <= UINT32_MAX
    assert len(body["statement_sha256"]) == 64
    assert all(ch in "0123456789abcdef" for ch in body["statement_sha256"])
    # Statements must never carry raw features, per-feature encodings, or proofs.
    leaked = {"temperature", "vibration", "current", "runtime", "features",
              "q_t", "q_v", "q_c", "q_r", "witness", "proof"}
    assert not (leaked & set(body))


def test_sample_q_score_matches_exact_rational_spec():
    body = _statement_body(SAMPLE)
    # q_* = roundTiesToEven(value*65536): 13107.2->13107, 19660.8->19661,
    # 26214.4->26214, 32768.0->32768.
    q_t, q_v, q_c, q_r = 13107, 19661, 26214, 32768
    expected = _round_ties_to_even(
        Fraction(q_t + q_v + q_c, 4) + Fraction(3 * q_r, 20) + Fraction(SCALE, 10)
    )
    assert body["q_score"] == expected == 26214


@pytest.mark.parametrize(
    "features,q_score",
    [
        ([0.0, 0.0, 0.0, 0.0], 6554),       # 6553.6 rounds up
        ([1.0, 1.0, 1.0, 1.0], 65536),      # exact 1.0
        ([0.2, 0.3, 0.4, 0.5], 26214),
    ],
)
def test_q_score_known_values(features, q_score):
    payload = {"model_id": "device-health-v1",
               "features": dict(zip(["temperature", "vibration", "current", "runtime"], features))}
    assert _statement_body(payload)["q_score"] == q_score


def test_feature_encoding_round_ties_to_even():
    # value * 65536 == 0.5 -> neighbor 0 (even); == 1.5 -> neighbor 2 (even).
    assert _quantize_feature(1 / 131072) == 0
    assert _quantize_feature(3 / 131072) == 2
    assert _quantize_feature(0.0) == 0
    assert _quantize_feature(1.0) == 65536
    assert _round_ties_to_even(Fraction(5, 2)) == 2
    assert _round_ties_to_even(Fraction(7, 2)) == 4


def test_quantized_value_within_error_bound_of_score_endpoint():
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1 / 131072, 3 / 131072, 0.999999, 0.123456]
    for t in grid:
        for r in (0.0, 0.5, 1.0):
            payload = {"model_id": "device-health-v1",
                       "features": {"temperature": t, "vibration": t,
                                    "current": t, "runtime": r}}
            statement = _statement_body(payload)
            score_result = client.post("/score", json=payload).json()
            assert abs(statement["q_score"] / SCALE - score_result["score"]) <= 1 / SCALE


def test_statement_sha256_is_jcs_over_all_other_fields():
    body = _statement_body(SAMPLE)
    preimage = {key: value for key, value in body.items() if key != "statement_sha256"}
    # Independent RFC 8785 style canonicalization: UTF-16-sorted keys, compact
    # JSON; all members here are ASCII strings or integers.
    canonical = json.dumps(preimage, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert body["statement_sha256"] == expected


def test_statement_sha256_deterministic_and_cross_process(tmp_path):
    first = _statement_body(SAMPLE)
    second = _statement_body(SAMPLE)
    assert first == second
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "device_proof", "statement", "--input", str(input_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert len(completed.stdout.strip().splitlines()) == 1
    cli_body = json.loads(completed.stdout)
    assert cli_body == first
    assert completed.stderr == ""


@pytest.mark.parametrize("value", [-0.1, 1.01])
def test_out_of_range_rejected_with_422(value):
    payload = copy.deepcopy(SAMPLE)
    payload["features"]["temperature"] = value
    assert client.post("/statements", json=payload).status_code == 422


def test_missing_feature_malformed_json_and_unknown_model():
    payload = copy.deepcopy(SAMPLE)
    del payload["features"]["runtime"]
    assert client.post("/statements", json=payload).status_code == 422
    assert client.post("/statements", content="{not json",
                       headers={"Content-Type": "application/json"}).status_code == 422
    payload = copy.deepcopy(SAMPLE)
    payload["model_id"] = "absent"
    response = client.post("/statements", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown model"


def test_statement_blocked_when_audit_fails(tmp_path, monkeypatch):
    from device_proof import audit
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads((ROOT / "models" / "device-health-v1.json").read_text())
    manifest["version"] = "2.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(audit, "MANIFEST_PATH", manifest_path)
    response = client.post("/statements", json=SAMPLE)
    assert response.status_code == 503
    assert response.json()["detail"] == "model_unavailable"


def test_cli_statement_ok(capsys):
    input_path = ROOT / "examples" / "sample.json"
    __main__.main(["statement", "--input", str(input_path)])
    out, err = capsys.readouterr()
    assert err == ""
    assert len(out.strip().splitlines()) == 1
    body = json.loads(out)
    assert set(body) == STATEMENT_KEYS
    assert body["q_score"] == 26214


@pytest.mark.parametrize(
    "setup,code",
    [
        (None, "input_unreadable"),
        (lambda p: p.write_text("{not json", encoding="utf-8"), "invalid_request"),
        (lambda p: p.write_text(json.dumps({"model_id": "absent",
                                            "features": {"temperature": 0.2, "vibration": 0.3,
                                                         "current": 0.4, "runtime": 0.5}}),
                                encoding="utf-8"), "unknown_model"),
        (lambda p: p.write_text(json.dumps({"model_id": "device-health-v1",
                                            "features": {"temperature": 2.0, "vibration": 0.3,
                                                         "current": 0.4, "runtime": 0.5}}),
                                encoding="utf-8"), "invalid_request"),
    ],
)
def test_cli_statement_failures(tmp_path, capsys, setup, code):
    input_path = tmp_path / "input.json"
    if setup is not None:
        setup(input_path)
    with pytest.raises(SystemExit) as excinfo:
        __main__.main(["statement", "--input", str(input_path)])
    assert excinfo.value.code != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == code


def test_encoding_overflow_rejected():
    # Unreachable through the validated HTTP surface (features are bounded),
    # but the encoder itself refuses values outside the uint32 domain.
    with pytest.raises(QuantizationError) as excinfo:
        _quantize_feature((UINT32_MAX + 1) / SCALE)
    assert excinfo.value.code == "encoding_out_of_uint32"


def test_module_does_not_export_or_claim_zk():
    import device_proof.statements as statements
    source = statements.__file__
    text = open(source, encoding="utf-8").read()
    assert "zero-knowledge" in text  # the explicit no-ZK statement
    assert not hasattr(statements, "witness")
    # The builder never returns raw inputs or encoded per-feature values.
    request = ScoreRequest.model_validate(SAMPLE)
    result = build_statement(request)
    assert set(result.model_dump()) == STATEMENT_KEYS
    assert score(ScoreRequest.model_validate(SAMPLE)).model_dump()["mode"] == "ordinary-inference"
