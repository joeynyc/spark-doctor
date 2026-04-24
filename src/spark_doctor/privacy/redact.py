from __future__ import annotations

import getpass
import os
import re
import socket
from typing import Any

try:
    from ..models import ScanReport
except ImportError:  # pragma: no cover
    ScanReport = None  # type: ignore


TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Hugging Face tokens
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<redacted:hf_token>"),
    # NGC API keys (base64-ish, long)
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b"), "<redacted:ngc_token>"),
    # Generic bearer / authorization
    (re.compile(r"(?i)\b(Bearer|Token)\s+[A-Za-z0-9_\-\.=]{12,}"), r"\1 <redacted:token>"),
    # api_key=... / apikey: ...
    (re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.=]{8,}['\"]?"),
     r"\1=<redacted:secret>"),
    # AWS-style
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted:aws_key>"),
    # JWT
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "<redacted:jwt>"),
    # OpenAI sk- keys
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "<redacted:sk_token>"),
    # SSH private key blocks
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "<redacted:private_key>"),
]

# Private IPv4 ranges (used when network identifiers are not included)
PRIVATE_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
)

MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")


def _current_identifiers() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        out["user"] = getpass.getuser()
    except Exception:  # noqa: BLE001
        pass
    try:
        out["host"] = socket.gethostname()
    except Exception:  # noqa: BLE001
        pass
    out["home"] = os.path.expanduser("~")
    return out


def redact_text(
    text: str,
    *,
    include_network_identifiers: bool = False,
    identifiers: dict[str, str] | None = None,
) -> str:
    if not isinstance(text, str) or not text:
        return text
    ids = identifiers if identifiers is not None else _current_identifiers()

    # Home directory path first (longer string)
    home = ids.get("home")
    if home and home not in ("/", ""):
        text = text.replace(home, "<redacted:home>")
    user = ids.get("user")
    if user and len(user) >= 2:
        text = re.sub(rf"\b{re.escape(user)}\b", "<redacted:user>", text)
    host = ids.get("host")
    if host and len(host) >= 2:
        text = re.sub(rf"\b{re.escape(host)}\b", "<redacted:host>", text)

    for pat, repl in TOKEN_PATTERNS:
        text = pat.sub(repl, text)

    if not include_network_identifiers:
        text = PRIVATE_IP_RE.sub("<redacted:private_ip>", text)
        text = MAC_RE.sub("<redacted:mac>", text)

    return text


def redact_obj(
    obj: Any,
    *,
    include_network_identifiers: bool = False,
    identifiers: dict[str, str] | None = None,
) -> Any:
    if identifiers is None:
        identifiers = _current_identifiers()
    if isinstance(obj, str):
        return redact_text(
            obj,
            include_network_identifiers=include_network_identifiers,
            identifiers=identifiers,
        )
    if isinstance(obj, list):
        return [
            redact_obj(x, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            for x in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            redact_obj(x, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            for x in obj
        )
    if isinstance(obj, dict):
        return {
            k: redact_obj(v, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            for k, v in obj.items()
        }
    return obj


def redact_report(report, *, include_network_identifiers: bool = False):
    """Return a new ScanReport with redacted content."""
    from ..models import ScanReport as _SR

    data = report.model_dump(mode="json")
    redacted = redact_obj(data, include_network_identifiers=include_network_identifiers)
    redacted["anonymized"] = True
    return _SR.model_validate(redacted)
