"""Exercise subscription probing without inference or real credentials."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from deeptutor.services.llm.provider_core.claude_code_provider import _build_options
from deeptutor.services.settings.provider_probe import probe_provider


class ClaudeSubscriptionProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_blank_url_uses_cli_discovery(self):
        with patch(
            "deeptutor.services.llm.claude_sdk.discover_models",
            new=AsyncMock(return_value=[{"slug": "sonnet", "display_name": "Sonnet"}]),
        ) as discovery:
            result = await probe_provider("claude_code", "", None)
        self.assertEqual(result["status"], "connected")
        self.assertEqual(result["models"], [{"id": "claude-code/sonnet"}])
        discovery.assert_awaited_once_with()

    async def test_missing_sdk_is_not_a_url_or_authentication_error(self):
        with patch(
            "deeptutor.services.llm.claude_sdk.discover_models",
            new=AsyncMock(side_effect=ModuleNotFoundError("No module named claude_agent_sdk")),
        ):
            result = await probe_provider("claude_code", "", None)
        self.assertEqual(result["status"], "dependency_missing")
        self.assertEqual(result["dependency"], "claude-agent-sdk")

    async def test_failed_discovery_does_not_claim_fallback_models_are_connected(self):
        with patch(
            "deeptutor.services.llm.claude_sdk.discover_models",
            new=AsyncMock(side_effect=RuntimeError("private-secret in CLI startup error")),
        ):
            result = await probe_provider("claude_code", "", None)
        self.assertEqual(result, {"status": "unavailable", "models": []})
        self.assertNotIn("private-secret", str(result))

    async def test_cli_timeout_is_reported(self):
        with patch(
            "deeptutor.services.llm.claude_sdk.discover_models",
            new=AsyncMock(side_effect=TimeoutError()),
        ):
            result = await probe_provider("claude_code", "", None)
        self.assertEqual(result, {"status": "timeout", "models": []})

    async def test_provider_uses_the_same_native_cli_as_discovery(self):
        sdk = SimpleNamespace(ClaudeAgentOptions=lambda **kwargs: kwargs)
        with patch("shutil.which", return_value="/native/claude"):
            options = _build_options(sdk, "claude-code/default", "", None, None, cwd="/tmp")
        self.assertEqual(options.get("cli_path"), "/native/claude")
        self.assertEqual(options["setting_sources"], [])

    async def test_test_provider_endpoint_accepts_empty_credentials(self):
        from deeptutor.api.routers import settings

        catalog = SimpleNamespace(load=lambda: {})
        with (
            patch.object(settings, "_require_settings_admin"),
            patch.object(settings, "get_model_catalog_service", return_value=catalog),
            patch.object(settings, "get_settings_draft_service", return_value=catalog),
            patch(
                "deeptutor.services.llm.claude_sdk.discover_models",
                new=AsyncMock(return_value=[{"slug": "sonnet", "display_name": "Sonnet"}]),
            ),
        ):
            result = await settings.test_provider_connection(
                settings.ProviderProbePayload(binding="claude_code", base_url="", api_key=None)
            )
        self.assertEqual(result["status"], "connected")

    async def test_test_provider_endpoint_explains_the_missing_sdk(self):
        from fastapi import HTTPException

        from deeptutor.api.routers import settings

        catalog = SimpleNamespace(load=lambda: {})
        with (
            patch.object(settings, "_require_settings_admin"),
            patch.object(settings, "get_model_catalog_service", return_value=catalog),
            patch.object(settings, "get_settings_draft_service", return_value=catalog),
            patch(
                "deeptutor.services.llm.claude_sdk.discover_models",
                new=AsyncMock(side_effect=ModuleNotFoundError("No module named claude_agent_sdk")),
            ),
            self.assertRaises(HTTPException) as error,
        ):
            await settings.test_provider_connection(settings.ProviderProbePayload(binding="claude_code"))
        self.assertEqual(error.exception.status_code, 503)
        self.assertIn("claude-agent-sdk", error.exception.detail)
        self.assertIn("No provider URL or API key", error.exception.detail)


if __name__ == "__main__":
    unittest.main()
