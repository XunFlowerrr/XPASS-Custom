"""Thin wrapper over lightning_sdk.Studio.

Everything the orchestrator needs from the platform goes through here, so the
SDK's quirks stay in one file: machine types are looked up by name, remote paths
must be absolute, and every command carries its own PATH export.
"""
from __future__ import annotations

import shlex
import time
from pathlib import Path

from .config import FleetConfig


def _sdk():
    """Import lazily so `fleet.config` stays usable without the SDK installed."""
    try:
        from lightning_sdk import Machine, Studio
    except ImportError as e:  # pragma: no cover - depends on the fleet group
        raise SystemExit(
            "lightning-sdk is not installed. Run:  uv sync --group fleet"
        ) from e
    return Machine, Studio


def machine_type(name: str):
    """Resolve 'L4' / 'CPU' / 'A100' to the SDK's Machine constant."""
    Machine, _ = _sdk()
    try:
        return getattr(Machine, name.upper())
    except AttributeError as e:
        available = sorted(a for a in dir(Machine) if a.isupper())
        raise SystemExit(f"Unknown machine {name!r}. Available: {available}") from e


class StudioError(RuntimeError):
    pass


class StudioHandle:
    """One Studio, addressed by name."""

    def __init__(self, cfg: FleetConfig, name: str, *, create_ok: bool = False) -> None:
        self.cfg = cfg
        self.name = name
        self._create_ok = create_ok
        self._studio = None

    # ─── lifecycle ────────────────────────────────────────────────────────────

    @property
    def studio(self):
        if self._studio is None:
            _, Studio = _sdk()
            kwargs = dict(name=self.name, teamspace=self.cfg.teamspace,
                          create_ok=self._create_ok)
            if self.cfg.org:
                kwargs["org"] = self.cfg.org
            if self.cfg.user:
                kwargs["user"] = self.cfg.user
            self._studio = Studio(**kwargs)
        return self._studio

    def exists(self) -> bool:
        try:
            _ = self.studio.status
            return True
        except Exception:
            return False

    def status(self) -> str:
        try:
            return str(self.studio.status)
        except Exception as e:
            return f"unknown ({e})"

    def ensure_started(self, machine: str) -> None:
        """Start on `machine`, or switch if it is already up on the wrong one."""
        st = self.status().lower()
        if "running" in st:
            current = str(getattr(self.studio, "machine", "") or "")
            if machine.upper() not in current.upper():
                self.studio.switch_machine(machine_type(machine))
            return

        kwargs = {"machine": machine_type(machine)}
        if self.cfg.interruptible:
            kwargs["interruptible"] = True
        if self.cfg.max_runtime:
            kwargs["max_runtime"] = self.cfg.max_runtime
        self.studio.start(**kwargs)

        deadline = time.time() + self.cfg.start_timeout_s
        while time.time() < deadline:
            if "running" in self.status().lower():
                return
            time.sleep(10)
        raise StudioError(f"{self.name} did not reach running in "
                          f"{self.cfg.start_timeout_s}s (status={self.status()})")

    def stop(self, *, best_effort: bool = True) -> None:
        try:
            self.studio.stop()
        except Exception as e:
            if not best_effort:
                raise
            print(f"⚠️  could not stop {self.name}: {e}")

    def duplicate_to(self, target_name: str, machine: str) -> "StudioHandle":
        """Copy this Studio -- filesystem included -- into a new one.

        duplicate() defaults to Machine.CPU, so the target machine has to be
        passed explicitly or the fleet comes up as CPU boxes.
        """
        self.studio.duplicate(machine=machine_type(machine), name=target_name)
        return StudioHandle(self.cfg, target_name)

    # ─── execution ────────────────────────────────────────────────────────────

    def exec(self, script: str) -> tuple[str, int]:
        """Run a shell snippet in the repo directory with PATH set."""
        cmd = f"{self.cfg.remote_env()} {script}"
        try:
            out, code = self.studio.run_with_exit_code(cmd)
        except Exception as e:
            return (f"{e}", 1)
        return (out or "", int(code))

    def exec_detached(self, script: str) -> None:
        """Launch something that must outlive this session."""
        wrapped = (f"{self.cfg.remote_env()} nohup setsid bash -c {shlex.quote(script)} "
                   f"</dev/null >/dev/null 2>&1 & echo started")
        try:
            self.studio.run(wrapped)
        except Exception as e:
            raise StudioError(f"failed to launch on {self.name}: {e}") from e

    def read_file(self, remote_abs: str) -> str | None:
        """Contents of a remote file, or None when it does not exist."""
        out, code = self.exec(f"cat {shlex.quote(remote_abs)} 2>/dev/null")
        return out.strip() if code == 0 and out.strip() else None

    def tail(self, remote_abs: str, lines: int = 40) -> str:
        out, _ = self.exec(f"tail -n {lines} {shlex.quote(remote_abs)} 2>/dev/null")
        return out

    # ─── files ────────────────────────────────────────────────────────────────

    @staticmethod
    def _require_abs(remote: str) -> None:
        # Lightning's transfer does not expand "~"; a relative path silently
        # succeeds and puts the file nowhere.
        if not remote.startswith("/"):
            raise ValueError(f"remote path must be absolute, got {remote!r}")

    def push(self, local: Path, remote_abs: str) -> None:
        self._require_abs(remote_abs)
        self.studio.upload_file(str(local), remote_abs)

    def pull_folder(self, remote_abs: str, local: Path) -> None:
        self._require_abs(remote_abs)
        local.mkdir(parents=True, exist_ok=True)
        self.studio.download_folder(remote_abs, str(local))
