"""Deterministic Q16 quantization statements for device-health-v1.

A statement pins down an ordinary-inference score under the fixed-point
quantization ``device-health-v1-q16-v1``: each feature is encoded as
``roundTiesToEven(value * 65536)`` and the quantized score is computed
exactly over rationals. The per-feature encodings (q_t, q_v, q_c, q_r) are
plain local integers: they stay in process memory only and are never logged,
persisted, or returned. The statement itself never carries raw features,
encoded feature values, or witnesses.

This is ordinary inference bookkeeping. No zero-knowledge proofs are
generated or claimed anywhere in this module.
"""

from fractions import Fraction
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict

from .audit import MODEL_ID, audited_artifacts
from .scoring import ScoreRequest

QUANTIZATION_ID = "device-health-v1-q16-v1"
SCALE = 65536
ROUNDING = "ties-to-even"
UINT32_MAX = 0xFFFFFFFF


class QuantizationError(ValueError):
    """An encoding escaped its allowed integer domain. Carries a stable code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class StatementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_sha256: str
    quantization_id: str
    q_score: int
    scale: int
    rounding: str
    statement_sha256: str


def _round_ties_to_even(value: Fraction) -> int:
    """Round a non-negative rational to the nearest integer, ties to even."""
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = 2 * remainder
    if doubled > value.denominator or (doubled == value.denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


def _quantize_feature(value: float) -> int:
    # Fraction(float) keeps the submitted binary value exact, so ties are
    # decided on the real input rather than a decimal re-parsing.
    encoded = _round_ties_to_even(Fraction(value) * SCALE)
    if not 0 <= encoded <= UINT32_MAX:
        raise QuantizationError("encoding_out_of_uint32")
    return encoded


def _jcs_escape_string(value: str) -> str:
    out = ['"']
    for ch in value:
        codepoint = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif codepoint == 0x08:
            out.append("\\b")
        elif codepoint == 0x09:
            out.append("\\t")
        elif codepoint == 0x0A:
            out.append("\\n")
        elif codepoint == 0x0C:
            out.append("\\f")
        elif codepoint == 0x0D:
            out.append("\\r")
        elif codepoint < 0x20:
            out.append(f"\\u{codepoint:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _jcs_serialize(value: Any) -> str:
    """Serialize the statement body per RFC 8785 (JSON Canonicalization).

    The statement object only contains strings and integers, so number
    serialization is plain decimal integers; keys sort by UTF-16 code units
    and output is encoded as UTF-8 bytes for hashing.
    """
    if isinstance(value, dict):
        ordered = sorted(value, key=lambda key: key.encode("utf-16-le"))
        members = ",".join(
            _jcs_escape_string(key) + ":" + _jcs_serialize(value[key]) for key in ordered
        )
        return "{" + members + "}"
    if isinstance(value, list):
        return "[" + ",".join(_jcs_serialize(item) for item in value) + "]"
    if isinstance(value, str):
        return _jcs_escape_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise TypeError("statement bodies contain only strings and integers")


def build_statement(request: ScoreRequest) -> StatementResponse:
    """Build the quantized statement for a validated score request."""
    if request.model_id != MODEL_ID:
        raise KeyError(request.model_id)
    # The fixed-model audit gates statements just as it gates inference; its
    # manifest also pins the feature order used for the q_* encoding order.
    manifest, _session = audited_artifacts(MODEL_ID)
    feature_order = manifest["feature_order"]
    # Encoded feature values live only in these local integers.
    encoded = [_quantize_feature(getattr(request.features, name)) for name in feature_order]
    q_t, q_v, q_c, q_r = encoded
    exact_q_score = (
        Fraction(q_t + q_v + q_c, 4) + Fraction(3 * q_r, 20) + Fraction(SCALE, 10)
    )
    q_score = _round_ties_to_even(exact_q_score)
    if not 0 <= q_score <= UINT32_MAX:
        raise QuantizationError("score_out_of_uint32")
    body = {
        "model_id": MODEL_ID,
        "model_sha256": manifest["sha256"],
        "quantization_id": QUANTIZATION_ID,
        "q_score": q_score,
        "scale": SCALE,
        "rounding": ROUNDING,
    }
    statement_sha256 = sha256(_jcs_serialize(body).encode("utf-8")).hexdigest()
    return StatementResponse(statement_sha256=statement_sha256, **body)
