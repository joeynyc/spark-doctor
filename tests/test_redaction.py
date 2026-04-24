from spark_doctor.privacy import redact_text, redact_obj


IDS = {"user": "zerocool", "host": "sparkbox", "home": "/home/zerocool"}


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
