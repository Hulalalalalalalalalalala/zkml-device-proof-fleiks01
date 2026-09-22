import copy
from fractions import Fraction
from hashlib import sha256
import json
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from device_proof import __main__, audit
from device_proof.api import app
from device_proof.audit import MODEL_ID
from device_proof.scoring import ROOT
from device_proof.statement import (
    QUANTIZATION_ID,
    SCALE,
    QuantizationError,
    _encode,
    _round_ties_to_even,
    jcs_dumps,
    statement_digest,
)

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


def test_statement_endpoint_fields():
    response = client.post("/statements", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == STATEMENT_KEYS
    assert body["model_id"] == MODEL_ID
    assert body["model_sha256"] == client.get(f"/models/{MODEL_ID}").json()["sha256"]
    assert body["quantization_id"] == QUANTIZATION_ID == "device-health-v1-q16-v1"
    assert body["scale"] == 65536
    assert body["rounding"] == "ties-to-even"
    assert isinstance(body["q_score"], int) and 0 <= body["q_score"] <= 2**32 - 1
    digest = body["statement_sha256"]
    assert digest == digest.lower() and len(digest) == 64
    int(digest, 16)


def test_statement_sample_q_score():
    body = client.post("/statements", json=SAMPLE).json()
    # q = (13107, 19661, 26214, 32768); exact rational sum is 26214.3.
    assert body["q_score"] == 26214


def test_statement_digest_preimage_is_rfc8785_jcs():
    body = client.post("/statements", json=SAMPLE).json()
    preimage = {key: body[key] for key in STATEMENT_KEYS - {"statement_sha256"}}
    assert body["statement_sha256"] == sha256(
        jcs_dumps(preimage).encode("utf-8")
    ).hexdigest()
    assert body["statement_sha256"] == statement_digest(preimage)


def test_statement_digest_consistent_across_processes():
    local = client.post("/statements", json=SAMPLE).json()
    completed = subprocess.run(
        [sys.executable, "-m", "device_proof", "statement", "--input", "examples/sample.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    remote = json.loads(completed.stdout)
    assert remote == local
    assert remote["statement_sha256"] == local["statement_sha256"]


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_statement_boundaries(value):
    payload = copy.deepcopy(SAMPLE)
    payload["features"] = dict.fromkeys(payload["features"], value)
    body = client.post("/statements", json=payload).json()
    expected = 6554 if value == 0.0 else 65536
    assert body["q_score"] == expected


def test_statement_within_one_ulp_of_score():
    for features in (
        {"temperature": 0.2, "vibration": 0.3, "current": 0.4, "runtime": 0.5},
        {"temperature": 0.0, "vibration": 0.0, "current": 0.0, "runtime": 0.0},
        {"temperature": 1.0, "vibration": 1.0, "current": 1.0, "runtime": 1.0},
        {"temperature": 0.11, "vibration": 0.83, "current": 0.37, "runtime": 0.64},
    ):
        payload = {"model_id": MODEL_ID, "features": features}
        score = client.post("/score", json=payload).json()["score"]
        q_score = client.post("/statements", json=payload).json()["q_score"]
        assert abs(q_score / SCALE - score) <= 1 / SCALE


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0),
        (1.0, 65536),
        (0.5 / 65536, 0),  # tie at 0.5 rounds to even 0
        (1.5 / 65536, 2),  # tie at 1.5 rounds to even 2
        (2.5 / 65536, 2),  # tie at 2.5 rounds to even 2
        (0.2, 13107),
    ],
)
def test_encode_ties_to_even(value, expected):
    assert _encode(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (Fraction(1, 2), 0),
        (Fraction(3, 2), 2),
        (Fraction(5, 2), 2),
        (Fraction(-1, 2), 0),
        (Fraction(-3, 2), -2),
        (Fraction(7, 3), 2),
    ],
)
def test_round_ties_to_even_exact(value, expected):
    assert _round_ties_to_even(value) == expected


@pytest.mark.parametrize("value", [-0.1, 70000.0])
def test_encode_out_of_range_rejected(value):
    with pytest.raises(QuantizationError) as excinfo:
        _encode(value)
    assert excinfo.value.code == "quantization_out_of_range"


def test_statement_out_of_range_and_unknown_model():
    payload = copy.deepcopy(SAMPLE)
    payload["features"]["temperature"] = 1.01
    assert client.post("/statements", json=payload).status_code == 422
    payload = copy.deepcopy(SAMPLE)
    del payload["features"]["runtime"]
    assert client.post("/statements", json=payload).status_code == 422
    payload = copy.deepcopy(SAMPLE)
    payload["model_id"] = "absent"
    assert client.post("/statements", json=payload).status_code == 404


def test_statement_carries_no_features_or_encoded_values():
    body = client.post("/statements", json=SAMPLE).json()
    forbidden = {"features", "temperature", "vibration", "current", "runtime",
                 "q_t", "q_v", "q_c", "q_r", "witness", "proof"}
    assert forbidden.isdisjoint(body)
    raw = json.dumps(body)
    for feature in SAMPLE["features"].values():
        assert str(feature) not in raw


def test_statement_audit_failure_returns_503(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "models" / "device-health-v1.json").read_text())
    manifest["version"] = "2.0.0"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(audit, "MANIFEST_PATH", path)
    response = client.post("/statements", json=SAMPLE)
    assert response.status_code == 503
    assert response.json()["detail"] == "model_unavailable"


def test_jcs_canonicalization():
    assert jcs_dumps({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert jcs_dumps({"x": "a\"b\\c\nd"}) == '{"x":"a\\"b\\\\c\\nd"}'
    assert jcs_dumps({"x": "\u0000\u001f"}) == '{"x":"\\u0000\\u001f"}'
    assert jcs_dumps({"x": "é"}) == '{"x":"é"}'
    assert jcs_dumps({"a": [True, False, None]}) == '{"a":[true,false,null]}'


def test_statement_cli_ok(capsys):
    __main__.main(["statement", "--input", str(ROOT / "examples" / "sample.json")])
    out, err = capsys.readouterr()
    assert err == ""
    assert len(out.strip().splitlines()) == 1
    body = json.loads(out)
    assert set(body) == STATEMENT_KEYS
    assert body["q_score"] == 26214


def test_statement_cli_unknown_model(tmp_path, capsys):
    payload = copy.deepcopy(SAMPLE)
    payload["model_id"] = "absent"
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        __main__.main(["statement", "--input", str(path)])
    assert excinfo.value.code != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "unknown_model"


def test_statement_cli_invalid_input(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        __main__.main(["statement", "--input", str(path)])
    assert excinfo.value.code != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "invalid_input"
