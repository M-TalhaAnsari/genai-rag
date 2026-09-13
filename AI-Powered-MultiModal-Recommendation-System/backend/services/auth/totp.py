"""
backend/services/auth/totp.py

RFC 4226 (HOTP) / RFC 6238 (TOTP) — stdlib only (hmac/hashlib/base64/
struct/time). Hand-rolled deliberately: this is ~40 lines against a
frozen spec, and pulling in a new third-party dependency for
something this small and this security-sensitive just moves the
"did they implement the spec correctly" question onto a package this
codebase doesn't otherwise depend on, without actually saving much
code. Compatible with Google Authenticator, Authy, 1Password, etc.
"""

import base64
import hashlib
import hmac
import os
import struct
import time
from typing import Optional
from urllib.parse import quote

TOTP_STEP_SECONDS = 30
TOTP_DIGITS = 6
TOTP_WINDOW = 1  # accept the previous/next 30s step too, for clock drift


def generate_secret() -> str:
    """160 random bits, base32-encoded with padding stripped — the
    standard authenticator-app secret format."""
    return base64.b32encode(os.urandom(20)).decode("utf-8").rstrip("=")


def _hotp(secret_b32: str, counter: int) -> str:
    padded = secret_b32 + "=" * ((8 - len(secret_b32) % 8) % 8)
    key = base64.b32decode(padded)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    code = code_int % (10 ** TOTP_DIGITS)
    return str(code).zfill(TOTP_DIGITS)


def verify_totp(secret_b32: str, code: str, for_time: Optional[int] = None) -> bool:
    if not code or not code.isdigit() or len(code) != TOTP_DIGITS:
        return False
    t = for_time if for_time is not None else int(time.time())
    counter = t // TOTP_STEP_SECONDS
    for offset in range(-TOTP_WINDOW, TOTP_WINDOW + 1):
        if hmac.compare_digest(_hotp(secret_b32, counter + offset), code):
            return True
    return False


def provisioning_uri(secret_b32: str, email: str, issuer: str = "Connoisseur") -> str:
    """otpauth:// URI. Feed it to any client-side QR renderer (the
    frontend can turn this into a QR code, or just show it as a
    manual-entry secret) — this function only builds the string."""
    label = quote(f"{issuer}:{email}")
    return (
        f"otpauth://totp/{label}?secret={secret_b32}"
        f"&issuer={quote(issuer)}&digits={TOTP_DIGITS}&period={TOTP_STEP_SECONDS}"
    )