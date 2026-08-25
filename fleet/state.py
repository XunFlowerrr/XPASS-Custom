"""Durable fleet state.

The orchestrator runs on a laptop, so it will be interrupted. Training is
detached on the studios and does not care, but the orchestrator must be able to
re-attach without re-running anything -- so every transition is written through
to disk, atomically.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .config import CREDITS_PER_HOUR, FleetConfig


class Phase(str, Enum):
    PLANNED = "planned"
    PROVISIONING = "provisioning"
    BOOTSTRAPPING = "bootstrapping"
    RUNNING = "running"
    COLLECTING = "collecting"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


#: A machine in one of these is finished with its current fold.
TERMINAL = {Phase.DONE, Phase.FAILED, Phase.STOPPED}


@dataclass
class MachineState:
    name: str
    phase: str = Phase.PLANNED.value
    fold: int | None = None
    started_at: float | None = None      # when the machine was started (billing)
    finished_at: float | None = None
    attempts: int = 0
    rc: int | None = None
    last_error: str | None = None
    log_tail: str = ""
    folds_done: list[int] = field(default_factory=list)

    def elapsed_hours(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at) / 3600.0


class FleetState:
    def __init__(self, path: Path, machines: dict[str, MachineState],
                 pending: list[int], commit: str) -> None:
        self.path = path
        self.machines = machines
        self.pending = pending
        self.commit = commit

    # ─── persistence ──────────────────────────────────────────────────────────

    @classmethod
    def load_or_init(cls, cfg: FleetConfig) -> "FleetState":
        if cfg.state_path.exists():
            raw = json.loads(cfg.state_path.read_text())
            if raw.get("commit") != cfg.commit:
                raise SystemExit(
                    f"{cfg.state_path} is for commit {raw.get('commit')}, "
                    f"but this run pins {cfg.commit}. Finish or delete it first."
                )
            machines = {n: MachineState(**m) for n, m in raw["machines"].items()}
            return cls(cfg.state_path, machines, list(raw["pending"]), cfg.commit)

        machines = {n: MachineState(name=n) for n in cfg.clone_names()}
        state = cls(cfg.state_path, machines, list(cfg.folds), cfg.commit)
        # Persist immediately. Waiting for the first transition would leave a
        # crash between init and provisioning with no record of the run at all,
        # and would keep the commit guard above from ever firing.
        state.save()
        return state

    def save(self) -> None:
        payload = {
            "commit": self.commit,
            "updated": time.time(),
            "pending": self.pending,
            "machines": {n: asdict(m) for n, m in self.machines.items()},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)   # atomic: never a half-written state file

    # ─── transitions ──────────────────────────────────────────────────────────

    def set(self, name: str, phase: Phase | None = None, **fields) -> MachineState:
        m = self.machines[name]
        if phase is not None:
            m.phase = phase.value
        for k, v in fields.items():
            setattr(m, k, v)
        self.save()
        return m

    # ─── work stealing ────────────────────────────────────────────────────────
    # Folds are independent, so they are not pinned to machines. If one studio
    # never starts, the rest drain the queue and the run still completes.

    def claim_fold(self, name: str) -> int | None:
        if not self.pending:
            return None
        fold = self.pending.pop(0)
        self.set(name, fold=fold)
        return fold

    def release_fold(self, name: str) -> None:
        """Return a machine's fold to the queue after a failure."""
        m = self.machines[name]
        if m.fold is not None and m.fold not in self.pending:
            self.pending.insert(0, m.fold)
        self.set(name, fold=None)

    # ─── reporting ────────────────────────────────────────────────────────────

    def machine_hours(self) -> float:
        return sum(m.elapsed_hours() for m in self.machines.values())

    def credits(self) -> float:
        return self.machine_hours() * CREDITS_PER_HOUR

    def all_settled(self) -> bool:
        return not self.pending and all(
            Phase(m.phase) in TERMINAL for m in self.machines.values())
