from __future__ import annotations

import functools
import getpass
import ipaddress
import os
import re
import socket
from collections.abc import Callable
from typing import Any

from ..models import ScanReport


REDACTION_RE = re.compile(r"(<redacted:[a-z0-9_]+>)")
SECRET_NAMES = (
    r"(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|password|secret|token|(?:access|auth|hf)[_-]token|"
    r"(?:openai|ngc)[_-]api[_-]?key|client[_-]secret|aws[_-]secret[_-]access[_-]key)"
)
SHELL_WORD_PART = r'''(?:"(?:\\.|[^"\\])*"|'[^']*'|\\.|[^\s'"\\])'''
SECRET_RE = re.compile(
    rf"'(?P<single_prefix>(?:--)?{SECRET_NAMES}=)[^']*'{SHELL_WORD_PART}*"
    rf'|"(?P<double_prefix>(?:--)?{SECRET_NAMES}=)(?:\\.|[^"\\])*"{SHELL_WORD_PART}*'
    rf"|(?P<prefix>(?<![\w-])(?:(?:--)?{SECRET_NAMES}['\"]?[ \t]*[:=][ \t]*"
    rf"|--{SECRET_NAMES}[ \t]+(?!--))){SHELL_WORD_PART}+",
    re.IGNORECASE,
)
SECRET_KEY_RE = re.compile(rf"(?:{SECRET_NAMES}|authorization)", re.IGNORECASE)
HOME_PATH_RE = re.compile(r'''(?<![\w])/(?:home|Users)/[^/\s"'<>:;,()]+''')


TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Hugging Face tokens
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<redacted:hf_token>"),
    # NGC API keys (base64-ish, long)
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b"), "<redacted:ngc_token>"),
    # GitHub classic and fine-grained tokens
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"), "<redacted:github_token>"),
    # Generic bearer / authorization
    (re.compile(r"(?i)(?<![\w-])(Bearer|Token)\s+(?!--)[A-Za-z0-9_\-\.=]{12,}"), r"\1 <redacted:token>"),
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

# The lookbehind pins matches to the start of a scheme so long dotted runs are scanned once.
URL_USERINFO_RE = re.compile(
    r"(?<![\w+.-])([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@:'\"<>]*:[^\s/@'\"<>]*@"
)
USER_CREDENTIAL_RE = re.compile(
    rf"(?P<prefix>(?<![\w-])(?:-[uU](?:[ \t]+|=)?|--(?:proxy-)?user(?:[ \t]+|=))(?!-))"
    rf"(?P<value>{SHELL_WORD_PART}+)"
)
UID_GID_RE = re.compile(r"\d+:\d+")
# Stops at shell separators so a following `docker run -p 8000:8000` keeps its port mapping.
DOCKER_LOGIN_RE = re.compile(r"(?<![\w-])docker[ \t]+login\b[^\r\n;&|]*")
DOCKER_PASSWORD_RE = re.compile(
    rf"(?P<prefix>(?<![\w-])-p(?:[ \t]+|=)?)(?!-|<redacted:)(?P<value>{SHELL_WORD_PART}+)"
)
PASSWORD_STDIN_RE = re.compile(
    rf"(?P<prefix>(?<![\w-])(?:echo|printf)(?:[ \t]+-[A-Za-z]+)*[ \t]+)(?!<redacted:)"
    rf"(?P<value>{SHELL_WORD_PART}+)(?P<tail>[ \t]*\|[ \t]*(?:sudo[ \t]+)?docker[ \t]+login\b)"
)
EMAIL_RE = re.compile(
    r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])"
)

# Dotted quads glued to version syntax (v1.2.3.4, ==1.2.3.4, 1.2.3.4-1, 1.2.3.4+cu) are versions.
IPV4_CANDIDATE_RE = re.compile(r"(?<![\w.+\-])(?<!==)(?:\d{1,3}\.){3}\d{1,3}(?![\w+\-]|\.\d)")
VERSION_CONTEXT_RE = re.compile(r"\b(?:v|version)[ \t]*[:=]?[ \t]*$", re.IGNORECASE)
PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.IPv4Network(net) for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")
)
# `\w` covers `_`; `.` stays allowed on the right of a host so FQDNs such as `spark-1a2b.local` are caught.
IDENTIFIER_LEFT = r"(?<!\w)(?<!\w[-./])"
USER_RIGHT = r"(?!\w)(?![-./]\w)"
HOST_RIGHT = r"(?!\w)(?![-/]\w)"

MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
IPV6_CANDIDATE_RE = re.compile(
    r"(?<!\w)[0-9A-Fa-f:.]+(?:%[A-Za-z0-9_.~-]+)?"
)
RDMA_GUID_LABEL = r"(?:node|port|system[ _-]*image)[ _-]*guid"
HARDWARE_KEY_RE = re.compile(
    rf"(?:serial(?:[ _-]*number)?|(?:gpu[ _-]*)?uuid|{RDMA_GUID_LABEL})", re.IGNORECASE
)
HARDWARE_LINE_RE = re.compile(
    rf"^([ \t|+`?\-\u2500-\u257f]*(?:serial(?:[ _-]*number)?|uuid|{RDMA_GUID_LABEL})[ \t]*:[ \t]*)[^ \t\r\n][^\r\n]*",
    re.IGNORECASE | re.MULTILINE,
)
GPU_UUID_RE = re.compile(
    r"\b(?:MIG-)?(?:GPU-|MIG-)[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def _replace_unredacted(
    text: str, pattern: re.Pattern[str], replacement: str | Callable[[re.Match[str]], str],
) -> str:
    # Redaction runs again at export; existing placeholders must survive unchanged.
    return "".join(
        part if index % 2 else pattern.sub(replacement, part)
        for index, part in enumerate(REDACTION_RE.split(text))
    )


def _redact_ipv6(match: re.Match[str]) -> str:
    candidate = match.group().rstrip(".")
    if ":" not in candidate:
        return match.group()
    # Validate the boundary here so malformed long tokens cannot force regex backtracking.
    following = match.string[match.end():match.end() + 1]
    if following and (following.isalnum() or following == "_"):
        return match.group()
    # Try the complete address first; colons can also be label separators, ports, or prose.
    starts = (0, 1) if candidate.startswith(":") else (0,)
    ends = (len(candidate), len(candidate) - 1) if candidate.endswith(":") else (len(candidate),)
    for start in starts:
        for end in ends:
            address = candidate[start:end]
            _, separator, port = address.rpartition(":")
            host_ends = [end]
            if separator and port.isdecimal() and len(port) <= 5 and int(port) <= 65535:
                host_ends.append(end - len(port) - 1)
            for host_end in host_ends:
                try:
                    ipaddress.IPv6Address(candidate[start:host_end])
                except ValueError:
                    continue
                return match.group()[:start] + "<redacted:ipv6>" + match.group()[host_end:]
    return match.group()


def _redact_secret(match: re.Match[str]) -> str:
    for group, quote in (("single_prefix", "'"), ("double_prefix", '"'), ("prefix", "")):
        if (prefix := match.group(group)) is not None:
            return f"{quote}{prefix}<redacted:secret>{quote}"
    return match.group()


def _redact_user_credential(match: re.Match[str]) -> str:
    value = match.group("value")
    # `-u`/`--user` without a password, or a `uid:gid` pair (docker run -u 1000:1000), carries no secret.
    if ":" not in value or UID_GID_RE.fullmatch(value):
        return match.group()
    return f"{match.group('prefix')}<redacted:secret>"


def _redact_docker_login(match: re.Match[str]) -> str:
    return DOCKER_PASSWORD_RE.sub(r"\g<prefix><redacted:secret>", match.group())


def _redact_ipv4(match: re.Match[str]) -> str:
    try:
        address = ipaddress.IPv4Address(match.group())
    except ValueError:
        return match.group()
    # Bind-all, multicast, netmasks and broadcast identify no host and are useful diagnostics.
    if address.is_unspecified or address.is_multicast or address.is_reserved:
        return match.group()
    if VERSION_CONTEXT_RE.search(match.string, max(0, match.start() - 16), match.start()):
        return match.group()
    if any(address in network for network in PRIVATE_IPV4_NETWORKS):
        return "<redacted:private_ip>"
    return "<redacted:ip>"


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

    # One pass keeps closing shell quotes and embedded placeholders inside their value.
    text = SECRET_RE.sub(_redact_secret, text)
    text = DOCKER_LOGIN_RE.sub(_redact_docker_login, text)
    text = PASSWORD_STDIN_RE.sub(r"\g<prefix><redacted:secret>\g<tail>", text)
    text = _replace_unredacted(text, URL_USERINFO_RE, r"\1<redacted:credentials>@")
    text = _replace_unredacted(text, USER_CREDENTIAL_RE, _redact_user_credential)
    for pattern, replacement in TOKEN_PATTERNS:
        text = _replace_unredacted(text, pattern, replacement)
    text = _replace_unredacted(text, HARDWARE_LINE_RE, r"\1<redacted:hardware_id>")
    text = _replace_unredacted(text, GPU_UUID_RE, "<redacted:hardware_id>")
    text = _replace_unredacted(text, EMAIL_RE, "<redacted:email>")

    home = ids.get("home")
    if home and home not in ("/", ""):
        text = _replace_unredacted(text, re.compile(re.escape(home)), "<redacted:home>")
    text = _replace_unredacted(text, HOME_PATH_RE, "<redacted:home>")
    user = ids.get("user")
    if user and len(user) >= 2:
        pattern = re.compile(rf"{IDENTIFIER_LEFT}{re.escape(user)}{USER_RIGHT}")
        text = _replace_unredacted(text, pattern, "<redacted:user>")
    for host in (ids.get("host"), ids.get("source_host")):
        if host and len(host) >= 2:
            pattern = re.compile(rf"{IDENTIFIER_LEFT}{re.escape(host)}{HOST_RIGHT}")
            text = _replace_unredacted(text, pattern, "<redacted:host>")

    if not include_network_identifiers:
        # IPv4-mapped and expanded IPv6 addresses overlap the IPv4/MAC patterns.
        text = _replace_unredacted(text, IPV6_CANDIDATE_RE, _redact_ipv6)
        text = _replace_unredacted(text, IPV4_CANDIDATE_RE, _redact_ipv4)
        text = _replace_unredacted(text, MAC_RE, "<redacted:mac>")

    return text


def redact_obj(
    obj: Any,
    *,
    include_network_identifiers: bool = False,
    identifiers: dict[str, str] | None = None,
) -> Any:
    if identifiers is None:
        identifiers = _current_identifiers()
    return _redact_obj(obj, include_network_identifiers, identifiers, frozenset())


def _redact_obj(
    obj: Any, include_network_identifiers: bool, identifiers: dict[str, str], schema_keys: frozenset[str],
) -> Any:
    if isinstance(obj, str):
        return redact_text(
            obj,
            include_network_identifiers=include_network_identifiers,
            identifiers=identifiers,
        )
    if isinstance(obj, (list, tuple)):
        items = (_redact_obj(x, include_network_identifiers, identifiers, schema_keys) for x in obj)
        return list(items) if isinstance(obj, list) else tuple(items)
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and SECRET_KEY_RE.fullmatch(k) and v is not None:
                value: Any = "<redacted:secret>"
            elif isinstance(k, str) and HARDWARE_KEY_RE.fullmatch(k) and v is not None:
                value = "<redacted:hardware_id>"
            else:
                value = _redact_obj(v, include_network_identifiers, identifiers, schema_keys)
            if isinstance(k, str) and k not in schema_keys:
                k = redact_text(k, include_network_identifiers=include_network_identifiers, identifiers=identifiers)
            # Distinct keys can redact to the same placeholder; the last one wins.
            out[k] = value
        return out
    return obj


@functools.cache
def _report_schema_keys() -> frozenset[str]:
    # A username or hostname equal to a field name (e.g. `docker`) must not rename report fields.
    schema = ScanReport.model_json_schema()
    models = [schema, *schema.get("$defs", {}).values()]
    return frozenset(name for model in models for name in model.get("properties", {}))


def redact_report(report: ScanReport, *, include_network_identifiers: bool = False) -> ScanReport:
    """Return a new ScanReport with redacted content."""
    data = report.model_dump(mode="json")
    identifiers = _current_identifiers()
    uname = report.os.get("uname")
    if isinstance(uname, str):
        parts = uname.split()
        if len(parts) >= 3 and parts[0] in {"Linux", "Darwin"} and not REDACTION_RE.fullmatch(parts[1]):
            identifiers["source_host"] = parts[1]
    redacted = _redact_obj(data, include_network_identifiers, identifiers, _report_schema_keys())
    redacted["anonymized"] = True
    return ScanReport.model_validate(redacted)
