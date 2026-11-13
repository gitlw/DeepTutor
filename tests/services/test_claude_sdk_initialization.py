"""Regress the SDK's 100-message startup queue without spawning Claude."""

import asyncio
import importlib.util
import json
import unittest
from unittest.mock import patch


class StartupTransport:
    def __init__(self, count=150, require_permission=False, reply=None):
        self.count = count
        self.require_permission = require_permission
        self.request = asyncio.Event()
        self.permission_reply = asyncio.Event()
        self.closed = False
        self.writes = []
        self.reply = reply
        self.user_message = asyncio.Event()
        self.input_closed = asyncio.Event()

    async def connect(self):
        pass

    async def write(self, data):
        message = json.loads(data)
        self.writes.append(message)
        if message.get("request", {}).get("subtype") == "initialize":
            self.request_id = message["request_id"]
            self.request.set()
        elif message.get("response", {}).get("request_id") == "permission":
            self.permission_reply.set()
        elif message.get("type") == "user":
            self.user_message.set()

    async def read_messages(self):
        await self.request.wait()
        for index in range(self.count):
            yield {"type": "system", "subtype": "hook_started", "index": index}
        if self.require_permission:
            yield {
                "type": "control_request", "request_id": "permission",
                "request": {"subtype": "can_use_tool", "tool_name": "example", "input": {}},
            }
            await self.permission_reply.wait()
        yield {
            "type": "control_response",
            "response": {"request_id": self.request_id, "subtype": "success", "response": {
                "models": [{"value": "sonnet"}],
            }},
        }
        yield {"type": "system", "subtype": "init", "index": self.count}
        if self.reply is not None:
            await self.user_message.wait()
            yield {"type": "assistant", "message": {"model": "sonnet", "content": [
                {"type": "text", "text": self.reply},
            ]}}
            yield {"type": "result", "subtype": "success", "duration_ms": 1,
                   "duration_api_ms": 1, "is_error": False, "num_turns": 1,
                   "session_id": "test", "result": self.reply}
            await self.input_closed.wait()
            return
        await asyncio.Event().wait()

    async def close(self):
        self.closed = True

    async def end_input(self):
        self.input_closed.set()

    def is_ready(self):
        return not self.closed


@unittest.skipUnless(importlib.util.find_spec("claude_agent_sdk"), "optional Claude SDK required")
class ClaudeSDKInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_startup_initializes_and_preserves_every_event(self):
        from claude_agent_sdk._internal.query import Query
        from deeptutor.services.llm.claude_sdk import InitializationBufferTransport

        transport = StartupTransport()
        query = Query(InitializationBufferTransport(transport), True, initialize_timeout=0.5)
        await query.start()
        try:
            info = await query.initialize()
            self.assertEqual(info["models"], [{"value": "sonnet"}])
            events = []
            async with asyncio.timeout(1):
                async for event in query.receive_messages():
                    events.append(event["index"])
                    if len(events) == 151:
                        break
            self.assertEqual(events, list(range(151)))
        finally:
            await query.close()
            query.close_receive_stream()
        self.assertTrue(transport.closed)

    async def test_permission_requests_still_run_before_initialization(self):
        from claude_agent_sdk import PermissionResultDeny
        from claude_agent_sdk._internal.query import Query
        from deeptutor.services.llm.claude_sdk import InitializationBufferTransport

        calls = []

        async def permission(tool_name, tool_input, context):
            calls.append(tool_name)
            return PermissionResultDeny(message="Denied by test")

        transport = StartupTransport(require_permission=True)
        query = Query(InitializationBufferTransport(transport), True,
                      can_use_tool=permission, initialize_timeout=0.5)
        await query.start()
        try:
            await query.initialize()
            self.assertEqual(calls, ["example"])
            response = next(row["response"] for row in transport.writes
                            if row.get("type") == "control_response")
            self.assertEqual(response["response"]["behavior"], "deny")
        finally:
            await query.close()
            query.close_receive_stream()

    async def test_discovery_uses_buffer_with_real_sdk_client_without_inference(self):
        from deeptutor.services.llm.claude_sdk import InitializationBufferTransport, discover_models

        transport = StartupTransport()
        with patch("deeptutor.services.llm.claude_sdk.create_transport",
                   return_value=InitializationBufferTransport(transport)):
            models = await discover_models()
        self.assertEqual(models, [{"slug": "sonnet", "display_name": "sonnet"}])
        self.assertTrue(transport.closed)
        self.assertFalse(any(row.get("type") == "user" for row in transport.writes))

    async def test_factory_retains_cli_flags_and_stream_protocol(self):
        from claude_agent_sdk import ClaudeAgentOptions
        from deeptutor.services.llm.claude_sdk import create_transport

        options = ClaudeAgentOptions(cli_path="/native/claude", model="sonnet", tools=[],
                                     permission_mode="dontAsk", setting_sources=[], cwd="/tmp")
        transport = create_transport(options, "A user prompt")
        command = transport._transport._build_command()
        self.assertEqual(command[0], "/native/claude")
        self.assertEqual(command[command.index("--model") + 1], "sonnet")
        self.assertEqual(command[command.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(command[command.index("--input-format") + 1], "stream-json")
        self.assertIn("--setting-sources=", command)

    async def test_provider_answers_after_large_startup_in_both_modes(self):
        from deeptutor.services.llm.claude_sdk import InitializationBufferTransport
        from deeptutor.services.llm.provider_core.claude_code_provider import ClaudeCodeProvider

        for session_id in ("", "learning-session"):
            with self.subTest(session_id=session_id):
                transport = StartupTransport(reply="A simulated answer")
                provider = ClaudeCodeProvider()
                try:
                    with patch(
                        "deeptutor.services.llm.provider_core.claude_code_provider.create_transport",
                        return_value=InitializationBufferTransport(transport),
                    ):
                        async with asyncio.timeout(2):
                            response = await provider.chat(
                                messages=[{"role": "user", "content": "A simulated question"}],
                                deeptutor_session_id=session_id,
                            )
                    self.assertEqual(response.content, "A simulated answer")
                    self.assertTrue(transport.user_message.is_set())
                finally:
                    await provider.aclose()
                self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
