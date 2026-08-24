"""Every tunable for a fleet run, in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Lightning bills per machine-hour; an L4 is 0.48 credits/hour.
CREDITS_PER_HOUR = 0.48

# $HOME is the same absolute path on every Studio, and it is the only thing that
# survives a stop/start. scp and the SDK do NOT expand "~", so paths built from
# this must stay absolute.
STUDIO_HOME = "/teamspace/studios/this_studio"


@dataclass(frozen=True)
class FleetConfig:
    # ─── identity ─────────────────────────────────────────────────────────────
    source_studio: str = "xpass"
    teamspace: str = "inference-optimization-project"
    org: str | None = "xunpiu"
    user: str | None = None

    # ─── what to run ──────────────────────────────────────────────────────────
    commit: str = ""                    # required: exact SHA every machine pins
    folds: tuple[int, ...] = (1, 2, 3, 4, 5)
    shards_per_machine: int = 2         # measured 1.79x aggregate on one L4
    num_workers: int = 4                # 8 vCPU / 2 shards
    machines: int = 5                   # one fold each

    # ─── machines ─────────────────────────────────────────────────────────────
    worker_machine: str = "L4"
    # The source only clones, cleans and verifies -- all I/O, no GPU needed.
    source_machine: str = "CPU"
    interruptible: bool = False
    # Opt-in hard ceiling. Off by default: control is from the Mac only, so if
    # this process dies the machines keep billing until stopped by hand.
    max_runtime: str | None = None

    # ─── paths ────────────────────────────────────────────────────────────────
    repo_dir: str = f"{STUDIO_HOME}/XPASS-Custom"
    status_dir: str = f"{STUDIO_HOME}/fleet/status"
    rclone_conf: str = f"{STUDIO_HOME}/.config/rclone/rclone.conf"
    state_path: Path = Path("fleet_state.json")
    collect_dir: Path = Path("fleet_results")

    # ─── behaviour ────────────────────────────────────────────────────────────
    poll_seconds: int = 60
    start_timeout_s: int = 900
    exec_timeout_s: int = 1800
    max_attempts: int = 2
    name_prefix: str = "xpass-f"

    def clone_names(self) -> list[str]:
        return [f"{self.name_prefix}{i}" for i in range(1, self.machines + 1)]

    def remote_env(self) -> str:
        """Prefix for every remote command.

        Studio.run() may not source .bashrc, and bootstrap installs both rclone
        and uv under $HOME, so the PATH has to be stated explicitly each time.
        """
        return f'export PATH="$HOME/bin:$HOME/.local/bin:$PATH"; cd {self.repo_dir};'


@dataclass
class MachineSpec:
    name: str
    fold: int | None = None
    tags: dict = field(default_factory=dict)
