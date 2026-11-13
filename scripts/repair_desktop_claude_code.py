"""Repair Claude Code provider support in an installed Open DeepTutor 1.7.2.

Run without flags to review the plan. Quit Open DeepTutor before --apply.
The repair installs the declared SDK and backs up four Python files before
replacing them. --check tests model discovery without inference; --restart
reopens the desktop with its local CLI search paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


BASE_HASHES = {
    "deeptutor/api/routers/settings.py":
        "3352177589a1a4082640ee04eaee849b672dffea8e10ec66a15e6a66117d51b9",
    "deeptutor/services/settings/provider_probe.py":
        "b399ca7bfb739946bf4eda0cb1e5ab1498a6505f9f462c747d0855313542200b",
    "deeptutor/services/llm/provider_core/claude_code_provider.py":
        "24126c8d33e039811bda3aebe6606fbb46fecbc7637c9c674081b2f68945518e",
    "deeptutor/services/llm/claude_sdk.py":
        "36bdeab398ed00980be7d53c8ffe5de1ccf764570e014e300f7ebb35059bcc32",
}
# Accept the first repair too, so users can apply this follow-up safely.
PREVIOUS_REPAIR_HASHES = {
    "deeptutor/services/llm/provider_core/claude_code_provider.py": {
        "4f1eefe44180360f8871d8ab90583ddcb8be58a4a70d5ca19e7cd7897c696a10",
        "b2250432f36898cc772566f3c9c350cde97663b28034c6479756dc77d45c77b9",
    },
    "deeptutor/services/llm/claude_sdk.py": {
        "4939ea3eb115deb31b428bc01add1d9b43579d67615c6c2f8fd9f233c98c169f",
    },
}
APP = "/Applications/Open DeepTutor.app"


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def cli_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PATH"] = ":".join(dict.fromkeys([
        str(Path.home() / ".local/bin"), "/opt/homebrew/bin", "/usr/local/bin",
        *environment.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin").split(":"),
    ]))
    return environment


def check_runtime(python: Path, runtime: Path) -> int:
    print("Checking Claude SDK initialization (up to 60 seconds; no model prompt)...", flush=True)
    code = """
import asyncio, json, time
from deeptutor.services.llm.claude_sdk import discover_models

async def check():
    start = time.monotonic()
    try:
        models = await discover_models()
        result = {"status": "connected" if models else "unavailable", "model_count": len(models)}
    except Exception as exc:
        result = {"status": "timeout" if isinstance(exc, TimeoutError) else "unavailable",
                  "error_type": type(exc).__name__}
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "connected" else 1

raise SystemExit(asyncio.run(check()))
"""
    environment = cli_environment()
    environment["PYTHONPATH"] = str(runtime / "core")
    try:
        result = subprocess.run(
            [str(python), "-c", code], cwd=runtime / "core", env=environment,
            timeout=80, check=False,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Claude initialization and cleanup exceeded 80 seconds.", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    desktop = Path.home() / "Library/Application Support/Open DeepTutor"
    active = json.loads((desktop / "active.json").read_text())
    runtime_id = active["runtimeId"]
    if active.get("coreVersion") != "1.7.2" or Path(runtime_id).name != runtime_id:
        raise RuntimeError("This repair supports the installed 1.7.2 runtime only.")
    runtime = desktop / "runtimes" / runtime_id
    python = runtime / "python/bin/python3"
    for name, original_hash in BASE_HASHES.items():
        replacement = source / name
        installed = runtime / "core" / name
        compile(replacement.read_text(), str(replacement), "exec")
        allowed = {original_hash, digest(replacement)} | PREVIOUS_REPAIR_HASHES.get(name, set())
        if digest(installed) not in allowed:
            raise RuntimeError(f"{name} differs from the supported build; nothing was changed.")

    with (source / "pyproject.toml").open("rb") as handle:
        requirements = tomllib.load(handle)["tool"]["deeptutor"]["optional-dependencies"][
            "claude-code"
        ]
    print(f"Installed runtime: {runtime_id}")
    print("Install:", ", ".join(requirements))
    print("Back up and repair:")
    for name in BASE_HASHES:
        print(f"  {name}")
    if not args.apply:
        if args.check:
            return check_runtime(python, runtime)
        print("Review only. Quit Open DeepTutor, then run with --apply --restart.")
        return 0

    commands = subprocess.check_output(["/bin/ps", "-axo", "command="], text=True)
    if any(
        line.strip() == APP + "/Contents/MacOS/Open DeepTutor"
        or line.strip().startswith(str(python) + " -m deeptutor_cli.main serve")
        for line in commands.splitlines()
    ):
        raise RuntimeError("Quit Open DeepTutor before installing dependencies or changing its runtime.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = desktop / "repair-backups" / stamp
    backup.mkdir(parents=True, mode=0o700)
    for name in BASE_HASHES:
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime / "core" / name, destination)
    print(f"Backup: {backup}")

    subprocess.run(
        # This is the app-owned bundled interpreter, marked as managed by uv.
        # Permit this explicit repair without changing the machine's Python.
        [str(python), "-m", "pip", "install", "--break-system-packages", "--no-input", *requirements],
        check=True,
    )
    subprocess.run([str(python), "-c", "import claude_agent_sdk"], check=True)
    for name in BASE_HASHES:
        installed = runtime / "core" / name
        staged = installed.with_suffix(".repair-tmp")
        shutil.copy2(source / name, staged)
        staged.replace(installed)
    print("Claude Code provider repair applied.")
    check_status = check_runtime(python, runtime) if args.check else 0
    if args.restart:
        cli_path = cli_environment()["PATH"]
        subprocess.run(["/usr/bin/open", "-a", APP, "--env", "PATH=" + cli_path], check=True)
        print("Desktop reopened. Retry Test provider with URL and API key blank.")
    return check_status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Repair stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
