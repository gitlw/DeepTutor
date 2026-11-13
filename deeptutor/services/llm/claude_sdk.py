"""Isolated, inference-free Claude SDK model discovery for both adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


CLAUDE_INITIALIZATION_TIMEOUT = 60


def local_cli_options() -> dict[str, Any]:
    """Find the installed CLI even when macOS Finder supplies a minimal PATH.

    The SDK prefers its bundled CLI when cli_path is omitted. That can change
    authentication and routing after a desktop restart. Preserve explicit PATH
    choices and use common macOS install directories as a fallback. Pass the
    extended PATH to the child too, for wrapper auth helpers and dependencies.
    """
    options: dict[str, Any] = {}
    cli = shutil.which("claude")
    if sys.platform == "darwin":
        inherited_path = os.environ.get("PATH", os.defpath)
        home = Path.home()
        search_path = os.pathsep.join(dict.fromkeys([
            *inherited_path.split(os.pathsep),
            str(home / ".local/bin"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            str(home / ".npm-global/bin"),
        ]))
        if cli is None:
            cli = shutil.which("claude", path=search_path)
        if search_path != inherited_path:
            options["env"] = {"PATH": search_path}
    if cli:
        options["cli_path"] = cli
    return options


class InitializationBufferTransport:
    """Keep startup events from blocking the SDK's initialize response.

    The SDK queues only 100 application messages and attaches their consumer
    after initialize completes. Plugin-rich CLIs can emit more than that
    before the handshake, deadlocking its reader. Buffer those events here
    until the initialize response passes through, then replay them in order.
    Control requests and cancellations always pass through immediately, so
    hooks and permission checks continue to run through the normal SDK.
    """

    def __init__(self, transport: Any):
        self._transport = transport
        self._initialize_id: str | None = None

    async def connect(self) -> None:
        await self._transport.connect()

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if not line.strip():
                continue
            message = json.loads(line)
            if (message.get("type") == "control_request"
                    and message.get("request", {}).get("subtype") == "initialize"):
                self._initialize_id = message["request_id"]
        await self._transport.write(data)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        initialized = False
        reader = self._transport.read_messages()
        try:
            async for message in reader:
                kind = message.get("type", "")
                if initialized:
                    yield message
                elif kind.startswith("control_"):
                    response = message.get("response", {})
                    if (kind == "control_response" and self._initialize_id is not None
                            and response.get("request_id") == self._initialize_id):
                        initialized = True
                        yield message
                        for event in pending:
                            yield event
                        pending.clear()
                    else:
                        yield message
                else:
                    pending.append(message)
            if not initialized:
                raise RuntimeError("Claude Code exited before SDK initialization completed.")
        finally:
            pending.clear()
            await reader.aclose()

    async def close(self) -> None:
        await self._transport.close()

    async def end_input(self) -> None:
        await self._transport.end_input()

    def is_ready(self) -> bool:
        return self._transport.is_ready()


def create_transport(options: Any, prompt: Any = None) -> InitializationBufferTransport:
    # Keep the SDK optional and use its own subprocess transport for flags,
    # authentication, parsing, stderr handling, and process cleanup.
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    async def empty_prompt():
        return
        yield {}

    return InitializationBufferTransport(SubprocessCLITransport(
        options=options, prompt=empty_prompt() if prompt is None else prompt,
    ))


def parse_models(rows: list) -> list[dict[str, str]]:
    models = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("value") or "").strip()
        if not value or value == "default" or row.get("disabled") or row.get("isDisabled"):
            continue
        if row.get("available") is False:
            continue
        slug = str(row.get("resolvedModel") or value).strip()
        if value.endswith("[1m]") and not slug.endswith("[1m]"):
            slug += "[1m]"
        if not slug or slug in seen:
            continue
        seen.add(slug)
        models.append({"slug": slug, "display_name": str(row.get("displayName") or slug)})
    return models


async def discover_models() -> list[dict[str, str]]:
    # Keep the SDK optional; CLI-only installs retain the existing fallback.
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options: dict[str, Any] = {
        **local_cli_options(),
        "tools": [],
        "setting_sources": [],
        "permission_mode": "dontAsk",
        "extra_args": {"strict-mcp-config": None},
    }
    # Isolate discovery from the repo whose settings page initiated sync.
    # Authentication and any wrapper-managed startup hooks stay CLI-managed.
    async with asyncio.timeout(CLAUDE_INITIALIZATION_TIMEOUT):
        return await _discover_in_directory(ClaudeAgentOptions, ClaudeSDKClient, options)


async def _discover_in_directory(ClaudeAgentOptions, ClaudeSDKClient, options):
    with tempfile.TemporaryDirectory(prefix="dt-claude-model-sdk-") as directory:
        options["cwd"] = directory
        sdk_options = ClaudeAgentOptions(**options)
        async with ClaudeSDKClient(
            options=sdk_options, transport=create_transport(sdk_options)
        ) as client:
            info = await client.get_server_info() or {}
            return parse_models(info.get("models") or [])
