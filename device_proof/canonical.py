"""RFC 8785 (JCS) JSON canonicalization for hashing proof material.

Proof material digests are computed over the JCS-canonical UTF-8 bytes of the
JSON document, so any party that holds the same logical JSON arrives at the
same SHA-256 digest regardless of key order or whitespace. Supported values:
objects with string keys, arrays, strings, integers, finite floats, booleans,
and null. Numbers follow ECMAScript Number::toString: integers in decimal,
floats via the shortest round-trip representation with ES exponent rules.
"""

from hashlib import sha256
import math
from typing import Any


def _escape_string(value: str) -> str:
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


def _es_number(value: float) -> str:
    """Serialize a finite float per ECMAScript Number::toString."""
    if not math.isfinite(value):
        raise TypeError("non-finite numbers are not valid JSON")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exponent = repr(abs(value)).partition("e")
    exp10 = int(exponent) if exponent else 0
    int_part, _, frac_part = mantissa.partition(".")
    digits = (int_part + frac_part).lstrip("0")
    exp10 -= len(frac_part)
    # value == int(digits) * 10**exp10; strip trailing zeros from the digits.
    stripped = digits.rstrip("0")
    exp10 += len(digits) - len(stripped)
    digits = stripped or "0"
    k = len(digits)
    n = exp10 + k  # value == 0.d1...dk * 10**n with d1 != 0
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        out = digits[0]
        if k > 1:
            out += "." + digits[1:]
        out += "e" + ("+" if n - 1 >= 0 else "-") + str(abs(n - 1))
    return sign + out


def canonical_json(value: Any) -> str:
    """Serialize a JSON value to its RFC 8785 canonical form."""
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("object keys must be strings")
        ordered = sorted(value, key=lambda key: key.encode("utf-16-le"))
        members = ",".join(
            _escape_string(key) + ":" + canonical_json(value[key]) for key in ordered
        )
        return "{" + members + "}"
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _es_number(value)
    if value is None:
        return "null"
    raise TypeError(f"unsupported value in canonical JSON: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()
