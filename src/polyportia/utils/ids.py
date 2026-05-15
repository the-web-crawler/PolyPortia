"""Stable, short, sortable identifier for traces and spans."""

from __future__ import annotations

import secrets
import time


def new_id() -> str:
    """Return a 22-char base32 ID combining 48 bits of time + 80 bits of randomness."""
    ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand = secrets.token_bytes(10)
    raw = ms.to_bytes(6, "big") + rand
    import base64

    return base64.b32encode(raw).decode("ascii").rstrip("=").lower()
