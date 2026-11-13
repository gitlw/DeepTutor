"""Finder launches must use the installed Claude login for both SDK paths."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deeptutor.services.llm import claude_sdk
from deeptutor.services.llm.provider_core.claude_code_provider import _build_options


class ClaudeDesktopCLITests(unittest.IsolatedAsyncioTestCase):
    async def test_finder_launch_uses_local_cli_for_provider_and_discovery(self):
        captured = []

        class Client:
            def __init__(self, options, transport=None):
                captured.append(options)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get_server_info(self):
                return {"models": [{"value": "sonnet"}]}

        sdk = SimpleNamespace(ClaudeAgentOptions=lambda **kwargs: kwargs, ClaudeSDKClient=Client)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cli = home / ".local/bin/claude"
            cli.parent.mkdir(parents=True)
            cli.write_text("#!/bin/sh\nexit 0\n")
            cli.chmod(0o700)
            with (
                patch.dict(os.environ, {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}),
                patch("pathlib.Path.home", return_value=home),
                patch("sys.platform", "darwin"),
                patch.dict("sys.modules", {"claude_agent_sdk": sdk}),
                patch.object(claude_sdk, "create_transport", return_value=object()),
            ):
                provider_options = _build_options(
                    sdk, "claude-code/claude-sonnet-5-5", "", None, None, cwd=directory
                )
                await claude_sdk.discover_models()
                for options in (provider_options, captured[0]):
                    self.assertEqual(options.get("cli_path"), str(cli))
                    # Wrapper dependencies such as auth helpers need the same PATH.
                    self.assertIn(str(cli.parent), options.get("env", {}).get("PATH", "").split(os.pathsep))
                    self.assertIn("/usr/local/bin", options["env"]["PATH"].split(os.pathsep))
                self.assertEqual(os.environ["PATH"], "/usr/bin:/bin:/usr/sbin:/sbin")

    async def test_existing_path_choice_takes_precedence(self):
        sdk = SimpleNamespace(ClaudeAgentOptions=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as directory:
            cli = Path(directory) / "claude"
            cli.write_text("#!/bin/sh\nexit 0\n")
            cli.chmod(0o700)
            with (
                patch.dict(os.environ, {"PATH": directory}),
                patch("sys.platform", "darwin"),
            ):
                options = _build_options(sdk, None, "", None, None, cwd=directory)
            self.assertEqual(options.get("cli_path"), str(cli))


if __name__ == "__main__":
    unittest.main()
