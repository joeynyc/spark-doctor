from __future__ import annotations

import shlex
import subprocess
import sys

import pytest

from spark_doctor.models import ProcessInfo, ScanReport
from spark_doctor.privacy import redact_text, redact_obj
from spark_doctor.privacy import redact_report


IDS = {"user": "zerocool", "host": "sparkbox", "home": "/home/zerocool"}


def test_unterminated_quoted_secret_does_not_stall_redaction():
    code = "from spark_doctor.privacy import redact_text; import sys; print(redact_text(sys.stdin.read()))"
    result = subprocess.run(
        [sys.executable, "-c", code], input="'token=" + "\\a" * 200,
        text=True, capture_output=True, timeout=3, check=True,
    )
    assert "<redacted:secret>" in result.stdout


def test_redacts_home_path():
    text = "loaded /home/zerocool/models/qwen.gguf"
    out = redact_text(text, identifiers=IDS)
    assert "/home/zerocool" not in out
    assert "<redacted:home>" in out


def test_redacts_hf_token():
    text = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456"
    out = redact_text(text, identifiers=IDS)
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in out


def test_redacts_bearer_token():
    text = "Authorization: Bearer abcdef1234567890xyz"
    out = redact_text(text, identifiers=IDS)
    assert "abcdef1234567890xyz" not in out
    assert "<redacted:token>" in out


def test_redacts_private_ip_by_default():
    text = "listening on 192.168.1.42"
    out = redact_text(text, identifiers=IDS)
    assert "192.168.1.42" not in out


def test_keeps_private_ip_when_allowed():
    text = "listening on 192.168.1.42"
    out = redact_text(text, include_network_identifiers=True, identifiers=IDS)
    assert "192.168.1.42" in out


def test_redacts_username_and_host():
    text = "user=zerocool host=sparkbox"
    out = redact_text(text, identifiers=IDS)
    assert "zerocool" not in out
    assert "sparkbox" not in out


def test_redact_obj_recursive():
    obj = {"cmd": "curl -H 'Authorization: Bearer secret_xyz_token_value'", "path": "/home/zerocool/x"}
    out = redact_obj(obj, identifiers=IDS)
    assert "secret_xyz_token_value" not in out["cmd"]
    assert "<redacted:home>" in out["path"]


def test_redacts_openai_key():
    text = "key=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out = redact_text(text, identifiers=IDS)
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in out


@pytest.mark.parametrize(
    "argument, secret",
    [
        ("--api-key private-value", "private-value"),
        ("--api_key private-value", "private-value"),
        ("--password=x!@:/a,b;c", "x!@:/a,b;c"),
        ("--password 'short secret'", "short secret"),
        ('--api-key="secret with spaces"', "secret with spaces"),
        ("--token abc", "abc"),
        ("--access-token private-value", "private-value"),
        ("PASSWORD=abc", "abc"),
        ('api_key: "a secret"', "a secret"),
        ("'--api-key=whole quoted credential'", "whole quoted credential"),
        ('"PASSWORD=whole quoted credential"', "whole quoted credential"),
    ],
)
def test_redacts_credential_arguments_and_assignments(argument: str, secret: str):
    text = f"serve {argument} --model qwen --max-tokens 4096"
    out = redact_text(text, identifiers=IDS)
    assert secret not in out
    assert "--model qwen --max-tokens 4096" in out
    assert "<redacted:" in out


def test_keeps_options_after_missing_credential_value():
    text = "serve --api-key --model qwen --token --context-length 4096"
    assert redact_text(text, identifiers=IDS) == text


def test_redacts_shell_quoted_password_containing_an_apostrophe():
    text = "serve --password 'first-part'\"'\"'last-part' --model qwen"
    out = redact_text(text, identifiers=IDS)
    assert out == "serve --password <redacted:secret> --model qwen"


@pytest.mark.parametrize("argument", ["'--api-key=whole quoted credential'", '"PASSWORD=whole quoted credential"'])
def test_redacts_entire_quoted_assignment(argument: str):
    out = redact_text(f"serve {argument} --model qwen", identifiers=IDS)
    assert "whole" not in out
    assert "quoted" not in out
    assert "credential" not in out
    assert "--model qwen" in out


@pytest.mark.parametrize("secret", [
    "two words", "first'last", 'a "quoted" value', "x!@:/a,b;c",
    "first last\\", "first\\'last", "<redacted:home>canary-secret",
])
@pytest.mark.parametrize("equals", [False, True])
def test_redacts_credentials_from_preserved_process_arguments(secret: str, equals: bool):
    credential = [f"--api-key={secret}"] if equals else ["--api-key", secret]
    arguments = shlex.join(["serve", *credential, "--model", "a model"])
    out = redact_text(arguments, identifiers=IDS)
    expected = ["--api-key=<redacted:secret>"] if equals else ["--api-key", "<redacted:secret>"]
    assert shlex.split(out) == ["serve", *expected, "--model", "a model"]
    assert redact_text(out, identifiers=IDS) == out


@pytest.mark.parametrize("prefix", ["--api-key=", "PASSWORD=", "token: "])
def test_explicit_assignment_redacts_values_starting_with_dashes(prefix: str):
    text = f"{prefix}--canary-secret --model qwen"
    out = redact_text(text, identifiers=IDS)
    assert "canary-secret" not in out
    assert "--model qwen" in out


@pytest.mark.parametrize("name", ["DB_PASSWORD", "WANDB_API_KEY", "ANTHROPIC_API_KEY", "custom-service-token"])
@pytest.mark.parametrize("form", ["env", "equals", "separate", "structured"])
def test_redacts_prefixed_credential_names(name: str, form: str):
    secret = "canary secret\\'tail"
    if form == "structured":
        out = redact_obj({name: secret, "tokenizer": "qwen", "max_tokens": 4096}, identifiers=IDS)
        assert out == {name: "<redacted:secret>", "tokenizer": "qwen", "max_tokens": 4096}
        return
    credentials = {
        "env": [f"{name}={secret}"],
        "equals": [f"--{name}={secret}"],
        "separate": [f"--{name}", secret],
    }[form]
    args = shlex.join(["serve", *credentials, "--model", "a model", "--max-tokens", "4096"])
    report = redact_report(ScanReport(processes=[ProcessInfo(pid=1, command="serve", args=args)]))
    assert "canary" not in report.model_dump_json()
    assert "tail" not in report.model_dump_json()
    expected = [part.replace(secret, "<redacted:secret>") for part in credentials]
    assert shlex.split(report.processes[0].args) == ["serve", *expected, "--model", "a model", "--max-tokens", "4096"]
    assert redact_report(report).model_dump() == report.model_dump()


def test_redacts_structured_credential_values_without_matching_ordinary_fields():
    obj = {
        "api_key": "short",
        "nested": {"Authorization": "Basic abc", "HF_TOKEN": "arbitrary"},
        "access-token": ["one", "two"],
        "tokenizer": "qwen",
        "max_tokens": 4096,
    }
    out = redact_obj(obj, identifiers=IDS)
    assert out["api_key"] == "<redacted:secret>"
    assert out["nested"] == {
        "Authorization": "<redacted:secret>", "HF_TOKEN": "<redacted:secret>"
    }
    assert out["access-token"] == "<redacted:secret>"
    assert out["tokenizer"] == "qwen"
    assert out["max_tokens"] == 4096
    assert obj["api_key"] == "short"


def test_redaction_is_idempotent_and_preserves_placeholders():
    text = "serve --api-key private-value --token hf_abcdefghijklmnopqrstuvwxyz123456"
    once = redact_text(text, identifiers=IDS)
    assert redact_text(once, identifiers=IDS) == once
    placeholders = "<redacted:home> <redacted:user> <redacted:host> <redacted:token>"
    assert redact_text(
        placeholders, identifiers={"user": "home", "host": "host", "home": "/home/home"}
    ) == placeholders


def test_imported_report_redacts_source_home_and_uname_hostname(monkeypatch):
    monkeypatch.setattr("spark_doctor.privacy.redact._current_identifiers", lambda: IDS)
    report = ScanReport(
        anonymized=False,
        os={"uname": "Linux source-spark 6.11.0 #1 SMP aarch64 GNU/Linux"},
        processes=[ProcessInfo(
            pid=1, command="serve", args="--api-key private-value --model /home/source-owner/qwen"
        )],
        reproduction_notes="source-spark loaded /Users/source-owner/work at 192.168.1.42",
    )
    result = redact_report(report)
    serialized = result.model_dump_json()
    assert "source-spark" not in serialized
    assert "source-owner" not in serialized
    assert "private-value" not in serialized
    assert "192.168.1.42" not in serialized
    assert result.anonymized
    assert result.os["uname"] == "Linux <redacted:host> 6.11.0 #1 SMP aarch64 GNU/Linux"
    assert "private-value" in report.processes[0].args
    assert redact_report(result).model_dump() == result.model_dump()
    with_network = redact_report(report, include_network_identifiers=True)
    assert "192.168.1.42" in with_network.reproduction_notes
    assert "private-value" not in with_network.model_dump_json()


@pytest.mark.parametrize("address", [
    "2001:db8:85a3::8a2e:370:7334", "2001:DB8:0:0:1:2:3:4", "fd12:3456::1",
    "fe80::1234:5678:abcd:ef01", "::1", "::", "::ffff:192.168.1.42",
    "::ffff:203.0.113.42", "fe80::1%eth0", "fe80::1%7", "fe80::1%25en0",
])
@pytest.mark.parametrize("template", [
    "eth0 UP {}/64", "http://[{}]:8000/v1", "peer:{}.", "Connection to {}: failed",
])
def test_redacts_ipv6_in_network_output_and_embedded_addresses(address: str, template: str) -> None:
    text = template.format(address)
    result = redact_text(text, identifiers=IDS)
    assert result == template.format("<redacted:ipv6>")
    assert redact_text(result, identifiers=IDS) == result
    assert redact_text(text, include_network_identifiers=True, identifiers=IDS) == text


@pytest.mark.parametrize("text", [
    "time=12:34:56", "PCI 0000:01:00.0", "cuda::device", "version 1.2.3",
    "invalid 2001:db8:12345::zzzz", "GPU temperature: 70", "aa:bb:cc:dd:ee:ff",
])
def test_ipv6_redaction_does_not_damage_non_addresses(text: str) -> None:
    expected = "<redacted:mac>" if text == "aa:bb:cc:dd:ee:ff" else text
    assert redact_text(text, identifiers=IDS) == expected


@pytest.mark.parametrize("address", [
    "2001:db8::5", "2001:db8:1:2:3:4:5:6", "::ffff:203.0.113.42",
])
@pytest.mark.parametrize("template", [
    "INFO {}:51234 - GET /v1/models", "peer {}:51234: failed", "IP6.ADDRESS[1]:{}",
    "IP6.ADDRESS[1]:{}:51234", "IP6.ADDRESS[1]:{}:51234: failed",
])
def test_redacts_ipv6_next_to_ports_and_label_separators(address: str, template: str) -> None:
    text = template.format(address)
    result = redact_text(text, identifiers=IDS)
    assert result == template.format("<redacted:ipv6>")
    assert redact_text(result, identifiers=IDS) == result
    assert redact_text(text, include_network_identifiers=True, identifiers=IDS) == text


def test_redacts_full_ipv6_address_with_short_unbracketed_port() -> None:
    text = "peer 2001:db8:1:2:3:4:5:6:22"
    assert redact_text(text, identifiers=IDS) == "peer <redacted:ipv6>:22"


@pytest.mark.parametrize("label", ["Serial Number", "Serial", "serial_number", "UUID"])
@pytest.mark.parametrize("prefix", ["\t", "│ │     ", "├─ ", "? ?   ", "?     "])
def test_redacts_firmware_identifier_lines_without_losing_other_fields(label: str, prefix: str) -> None:
    text = f"{prefix}{label}: TEST-SERIAL-1234\n{prefix}Current version: 1.2.3\n"
    result = redact_text(text, include_network_identifiers=True, identifiers=IDS)
    assert result == f"{prefix}{label}: <redacted:hardware_id>\n{prefix}Current version: 1.2.3\n"
    assert redact_text(result, identifiers=IDS) == result


def test_missing_firmware_serial_does_not_consume_next_line() -> None:
    text = "Serial Number:   \nCurrent version: 1.2.3\n"
    assert redact_text(text, identifiers=IDS) == text


def test_malformed_ipv6_candidate_does_not_stall_redaction() -> None:
    code = "from spark_doctor.privacy import redact_text; import sys; redact_text(sys.stdin.read())"
    subprocess.run(
        [sys.executable, "-c", code], input="a:" * 10_000 + "G",
        text=True, capture_output=True, timeout=3, check=True,
    )


@pytest.mark.parametrize("prefix", ["GPU-", "MIG-", "MIG-GPU-"])
def test_redacts_gpu_uuid_in_free_text_but_preserves_firmware_model_guid(prefix: str) -> None:
    uuid = "00000000-1111-2222-3333-444444444444"
    text = f"selected {prefix}{uuid}; GUID: {uuid}"
    result = redact_text(text, include_network_identifiers=True, identifiers=IDS)
    assert result == f"selected <redacted:hardware_id>; GUID: {uuid}"
    assert redact_text(result, identifiers=IDS) == result


@pytest.mark.parametrize("key", ["uuid", "gpu_uuid", "Serial Number", "serial_number", "serial", "SerialNumber"])
def test_redacts_structured_hardware_identifiers(key: str) -> None:
    original = {key: "TEST-SERIAL-1234", "empty": {key: None}, "version": "1.2.3"}
    redacted = redact_obj(original, include_network_identifiers=True, identifiers=IDS)
    assert redacted == {key: "<redacted:hardware_id>", "empty": {key: None}, "version": "1.2.3"}
    assert redact_obj(redacted, identifiers=IDS) == redacted
    assert original[key] == "TEST-SERIAL-1234"


def test_default_report_masks_real_collector_identifier_shapes() -> None:
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    report = ScanReport(
        gpu={"gpus": [{"uuid": uuid, "name": "NVIDIA GB10"}]},
        firmware={"fwupdmgr": "│ │ Serial Number: TEST-SERIAL-1234\n│ │ Current version: 1.2.3"},
        network={"ip_br_addr": "eth0 UP 2001:db8::1/64 fe80::1%eth0/64"},
    )
    result = redact_report(report)
    serialized = result.model_dump_json()
    assert all(identifier not in serialized for identifier in (uuid, "TEST-SERIAL-1234", "2001:db8::1", "fe80::1"))
    assert result.gpu["gpus"][0]["name"] == "NVIDIA GB10"
    assert "1.2.3" in result.firmware["fwupdmgr"]
    assert redact_report(result).model_dump() == result.model_dump()
    with_network = redact_report(report, include_network_identifiers=True).model_dump_json()
    assert "2001:db8::1" in with_network and "fe80::1%eth0" in with_network
    assert uuid not in with_network and "TEST-SERIAL-1234" not in with_network


@pytest.mark.parametrize("text, expected", [
    ("git clone https://joe:hunter2@github.com/x/y.git", "git clone https://<redacted:credentials>@github.com/x/y.git"),
    ("REDIS_URL=redis://:hunter2@10.0.0.2:6379/0", "REDIS_URL=redis://<redacted:credentials>@<redacted:private_ip>:6379/0"),
    ("curl -u admin:hunter2 http://x", "curl -u <redacted:secret> http://x"),
    ("curl --user admin:hunter2 http://x", "curl --user <redacted:secret> http://x"),
    ("docker login -p hunter2pass nvcr.io", "docker login -p <redacted:secret> nvcr.io"),
    ("docker login -u '$oauthtoken' --password hunter2pass nvcr.io",
     "docker login -u '$oauthtoken' --password <redacted:secret> nvcr.io"),
    ("echo hunter2pass | docker login -u x --password-stdin nvcr.io",
     "echo <redacted:secret> | docker login -u x --password-stdin nvcr.io"),
])
def test_redacts_inline_credentials(text: str, expected: str) -> None:
    assert redact_text(text, identifiers=IDS) == expected
    assert redact_text(expected, identifiers=IDS) == expected


@pytest.mark.parametrize("text", [
    "docker run -u 1000:1000 -p 8000:8000 vllm",
    "docker login --password-stdin nvcr.io && docker run -p 8000:8000 vllm",
    "pip install --user vllm; sudo -u root ls",
    "https://github.com/x/y.git ssh://github.com/x",
])
def test_keeps_user_flags_and_urls_without_credentials(text: str) -> None:
    assert redact_text(text, identifiers=IDS) == text


@pytest.mark.parametrize("token", [
    "ghp_" + "A1b2" * 9, "gho_" + "A1b2" * 9, "ghu_" + "A1b2" * 9, "ghs_" + "A1b2" * 9, "ghr_" + "A1b2" * 9,
    "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz0123456789",
])
def test_redacts_github_tokens(token: str) -> None:
    assert redact_text(f"GIT_AUTH {token} done", identifiers=IDS) == "GIT_AUTH <redacted:github_token> done"


def test_redacts_email_addresses() -> None:
    text = "contact joe.smith+spark@example.co.uk, now"
    assert redact_text(text, identifiers=IDS) == "contact <redacted:email>, now"


@pytest.mark.parametrize("address, placeholder", [
    ("8.8.8.8", "<redacted:ip>"),
    ("100.64.12.7", "<redacted:ip>"),
    ("169.254.3.4", "<redacted:ip>"),
    ("172.16.0.9", "<redacted:private_ip>"),
    ("127.0.0.1", "<redacted:private_ip>"),
])
@pytest.mark.parametrize("template", ["peer {}", "http://{}:8000/v1", "inet {}/24 brd", "--host={}", "at {}."])
def test_redacts_all_ipv4_addresses(address: str, placeholder: str, template: str) -> None:
    text = template.format(address)
    result = redact_text(text, identifiers=IDS)
    assert result == template.format(placeholder)
    assert redact_text(result, identifiers=IDS) == result
    assert redact_text(text, include_network_identifiers=True, identifiers=IDS) == text


@pytest.mark.parametrize("text", [
    "driver 580.95.05 torch 2.5.1 kernel 6.11.0-1016-nvidia CUDA 13.0",
    "pkg==1.2.3.4", "v1.2.3.4", "version 1.2.3.4", "Version: 1.2.3.4", "pkg-1.2.3.4", "lib_1.2.3.4",
    "1.2.3.4-1ubuntu1", "1.2.3.4+cu130", "1.2.3.4.5", "+1.2.3.4",
    "bind 0.0.0.0:8000", "netmask 255.255.255.0", "mdns 224.0.0.251",
])
def test_ipv4_redaction_keeps_versions_and_non_host_addresses(text: str) -> None:
    assert redact_text(text, identifiers=IDS) == text


def test_identifier_glued_into_longer_tokens_is_kept() -> None:
    ids = {"user": "nvidia", "host": "spark-1a2b", "home": "/home/nvidia"}
    text = "nvcr.io/nvidia/vllm nvidia-smi nvidia-container-runtime nvidia_peermem libnvidia.so spark-1a2b-old"
    assert redact_text(text, identifiers=ids) == text


def test_standalone_identifiers_still_redacted() -> None:
    ids = {"user": "nvidia", "host": "spark-1a2b", "home": "/home/nvidia"}
    text = "user nvidia. ssh nvidia@spark-1a2b or spark-1a2b.local; ls /home/nvidia/models /home/other/x"
    assert redact_text(text, identifiers=ids) == (
        "user <redacted:user>. ssh <redacted:user>@<redacted:host> or <redacted:host>.local;"
        " ls <redacted:home>/models <redacted:home>/x"
    )


def test_redact_obj_redacts_dictionary_keys() -> None:
    obj = {
        "hf_abcdefghijklmnopqrstuvwxyz": "log text",
        "/home/zerocool/app.log": {"uuid": "TEST-SERIAL-1234"},
        "uuid": "GPU-x", "python_executable": "/usr/bin/python3", "mem_total_kb": 1, 7: "int key",
    }
    out = redact_obj(obj, identifiers=IDS)
    assert out == {
        "<redacted:hf_token>": "log text",
        "<redacted:home>/app.log": {"uuid": "<redacted:hardware_id>"},
        "uuid": "<redacted:hardware_id>", "python_executable": "/usr/bin/python3", "mem_total_kb": 1, 7: "int key",
    }
    assert redact_obj(out, identifiers=IDS) == out


def test_redact_obj_key_collision_keeps_last_value() -> None:
    out = redact_obj({"user zerocool": "first", "user <redacted:user>": "second"}, identifiers=IDS)
    assert out == {"user <redacted:user>": "second"}


def test_report_keys_redacted_without_renaming_fields_matching_identifiers(monkeypatch) -> None:
    monkeypatch.setattr(
        "spark_doctor.privacy.redact._current_identifiers",
        lambda: {"user": "docker", "host": "name", "home": "/home/docker"},
    )
    report = ScanReport(
        docker={"version": "28.0"},
        logs={"hf_abcdefghijklmnopqrstuvwxyz": "boot ok", "journal docker": "user docker"},
        collector_statuses=[{"name": "gpu", "ok": True}],
    )
    result = redact_report(report)
    assert result.docker == {"version": "28.0"}
    assert result.collector_statuses[0].name == "gpu"
    assert result.logs == {"<redacted:hf_token>": "boot ok", "journal <redacted:user>": "user <redacted:user>"}
