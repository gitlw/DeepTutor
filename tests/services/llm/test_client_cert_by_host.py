"""mTLS client certificates are applied per destination host only."""

from __future__ import annotations

import ssl

import pytest

from deeptutor.services.llm import openai_http_client as mod
from deeptutor.services.llm.exceptions import LLMConfigError

GW = "https://gw.example.com/v1/projects/p"


def _settings(monkeypatch, **overrides):
    monkeypatch.setattr(
        mod, "load_system_settings", lambda: {"disable_ssl_verify": False, **overrides}
    )


def test_cert_applies_to_exact_host_only(monkeypatch):
    _settings(monkeypatch, client_cert_by_host={"gw.example.com": "/x/agent.pem"})
    assert mod.client_cert_for_url(GW) == "/x/agent.pem"
    assert mod.client_cert_for_url("https://other.example.com/v1") is None
    assert mod.tls_verify_for_url("https://other.example.com/v1") is True
    assert mod.aiohttp_request_options("https://other.example.com/v1") == {}


def test_missing_cert_file_is_a_config_error(monkeypatch, tmp_path):
    _settings(monkeypatch, client_cert_by_host={"gw.example.com": str(tmp_path / "nope.pem")})
    with pytest.raises(LLMConfigError, match="client certificate"):
        mod.tls_verify_for_url(GW)


def test_sdk_client_uses_mtls_context(monkeypatch):
    _settings(monkeypatch, client_cert_by_host={"gw.example.com": "/x/agent.pem"})
    loaded: list[str] = []
    monkeypatch.setattr(ssl.SSLContext, "load_cert_chain", lambda self, cert, *a, **k: loaded.append(cert))
    kwargs = mod.openai_sdk_client_kwargs(api_key="k", base_url=GW, disable_ssl_verify=False)
    assert loaded == ["/x/agent.pem"]
    assert "http_client" in kwargs
    assert isinstance(mod.aiohttp_request_options(GW)["ssl"], ssl.SSLContext)
