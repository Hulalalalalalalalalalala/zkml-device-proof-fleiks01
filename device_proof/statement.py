"""Quantized statements for the bundled demonstration model.

The statement path re-encodes the four validated features as Q16.16
fixed-point integers (roundTiesToEven(value * 65536), uint32) in manifest
feature order, combines them with exact rational arithmetic into a
quantized score, and commits to the public response fields with an
RFC 8785 (JCS) SHA-256 digest. Encoded values live only in memory for
the duration of the call; the statement itself carries no raw features,
no encoded values, and no witness. This is ordinary quantized
arithmetic: no zero-knowledge proof is generated or claimed anywhere on
this path.
"""

from fractions import Fraction
from hashlib import sha256

from pydantic import BaseModel, ConfigDict

from .audit import MODEL_ID, audited_artifacts
from .scoring import ScoreRequest

QUANTIZATION_ID = "device-health-v1-q16-v1"
SCALE = 65536
ROUNDING = "ties-to-even"
UINT32_MAX = 2**32 - 1


class QuantizationError(Exception):
    """Quantization failure carrying a stable, non-sensitive error code."""

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
    """Round an exact rational to the nearest integer, ties to even."""
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder < Fraction(1, 2):
        return floor
    if remainder > Fraction(1, 2):
        return floor + 1
    return floor if floor % 2 == 0 else floor + 1


def _encode(value: float) -> int:
    """Encode one validated feature as a uint32 Q16.16 integer."""
    quantized = _round_ties_to_even(Fraction(value) * SCALE)
    if not 0 <= quantized <= UINT32_MAX:
        raise QuantizationError("quantization_out_of_range")
    return quantized


def _q_score(q_t: int, q_v: int, q_c: int, q_r: int) -> int:
    """Combine the encoded features into the quantized score, exactly."""
    exact = (
        Fraction(q_t + q_v + q_c, 4)
        + Fraction(3 * q_r, 20)
        + Fraction(SCALE, 10)
    )
    q_score = _round_ties_to_even(exact)
    if not 0 <= q_score <= UINT32_MAX:
        raise QuantizationError("quantization_overflow")
    return q_score


_JCS_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}


def _jcs_string(value: str) -> str:
    out = ['"']
    for char in value:
        code = ord(char)
        if code in _JCS_ESCAPES:
            out.append(_JCS_ESCAPES[code])
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def jcs_dumps(value) -> str:
    """Serialize a JSON value per RFC 8785 (JCS), deterministically."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(jcs_dumps(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(
            f"{_jcs_string(key)}:{jcs_dumps(value[key])}" for key in keys
        ) + "}"
    raise TypeError(f"unsupported type for JCS: {type(value)!r}")


def statement_digest(fields: dict) -> str:
    """SHA-256 of the JCS-canonicalized statement fields, lowercase hex."""
    return sha256(jcs_dumps(fields).encode("utf-8")).hexdigest()


def statement(request: ScoreRequest) -> StatementResponse:
    """Build the quantized statement for a validated score request."""
    if request.model_id != MODEL_ID:
        raise KeyError(request.model_id)
    manifest, _session = audited_artifacts(MODEL_ID)
    feature_order = manifest["feature_order"]
    # Encoded values are used only within this call and never persisted
    # or included in the statement.
    q_t, q_v, q_c, q_r = (
        _encode(getattr(request.features, feature)) for feature in feature_order
    )
    q_score = _q_score(q_t, q_v, q_c, q_r)
    fields = {
        "model_id": MODEL_ID,
        "model_sha256": manifest["sha256"],
        "quantization_id": QUANTIZATION_ID,
        "q_score": q_score,
        "scale": SCALE,
        "rounding": ROUNDING,
    }
    return StatementResponse(**fields, statement_sha256=statement_digest(fields))
