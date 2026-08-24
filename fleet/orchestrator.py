"""Drive a fleet of Studios through the finetune sweep.

Shape of a run:

    prepare_source()   clean + pin + verify the golden studio, then stop it
    provision()        duplicate it once per machine, on L4
    bootstrap()        per machine, idempotent, tolerates the ephemeral root
    dispatch()         detached run_fold.sh; the machine now runs unattended
    monitor()          poll sentinels; stop each machine the moment it finishes
    finalize()         aggregate locally -- no GPU, no credits

Control is from this process only. Nothing on the studios stops them, so if this
dies the machines keep billing; `fleet down` is the manual backstop.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from .config import CREDITS_PER_HOUR, FleetConfig
from .state import FleetState, Phase
from .studios import StudioError, StudioHandle


class Orchestrator:
    def __init__(self, cfg: FleetConfig) -> None:
        if not cfg.commit:
            raise SystemExit("FleetConfig.commit is required -- pin an exact SHA.")
        self.cfg = cfg
        self.state = FleetState.load_or_init(cfg)
        self.handles: dict[str, StudioHandle] = {
            n: StudioHandle(cfg, n) for n in cfg.clone_names()
        }

    # ─── source studio ────────────────────────────────────────────────────────

    def prepare_source(self) -> None:
        """Get the golden studio into the state we want every clone to inherit.

        Runs on CPU: this is disk and git work, no GPU. Stopped at the end,
        because duplicating a running studio risks a mid-write snapshot.
        """
        cfg = self.cfg
        src = StudioHandle(cfg, cfg.source_studio)
        if not src.exists():
            raise SystemExit(f"Source studio {cfg.source_studio!r} not found in "
                             f"teamspace {cfg.teamspace!r}")

        print(f"▶ starting {cfg.source_studio} on {cfg.source_machine}")
        src.ensure_started(cfg.source_machine)

        print("▶ bootstrapping source (clean, pin, verify)")
        out, code = src.exec(f"bash scripts/bootstrap_studio.sh {shlex.quote(cfg.commit)}")
        print(out)
        if code != 0:
            raise SystemExit(
                f"source bootstrap failed (exit {code}). Fix it before duplicating "
                f"-- every clone inherits this filesystem."
            )

        used, _ = src.exec("du -sh $HOME 2>/dev/null | cut -f1")
        print(f"▶ source home is {used.strip()}; stopping before duplication")
        src.stop(best_effort=False)

    # ─── provisioning ─────────────────────────────────────────────────────────

    def provision(self) -> None:
        """Duplicate the source once per machine, serially.

        duplicate() carries the filesystem, so Dataset/ and models_pth/ come
        along and no clone re-downloads 17 GB from Google Drive -- which also
        avoids the rate-limit interstitial that setup_data.sh has to defend
        against. Serial because a burst of 17 GB copies invites throttling and
        gives no clean failure boundary.
        """
        cfg = self.cfg
        src = StudioHandle(cfg, cfg.source_studio)

        for name in cfg.clone_names():
            m = self.state.machines[name]
            if Phase(m.phase) not in (Phase.PLANNED, Phase.FAILED):
                continue
            h = self.handles[name]
            if h.exists():
                print(f"▶ {name} already exists, reusing")
                self.state.set(name, Phase.PROVISIONING)
                continue

            print(f"▶ duplicating {cfg.source_studio} -> {name} ({cfg.worker_machine})")
            self.state.set(name, Phase.PROVISIONING)
            try:
                src.duplicate_to(name, cfg.worker_machine)
                self.state.set(name, Phase.PROVISIONING, last_error=None)
            except Exception as e:
                print(f"⚠️  duplicate failed for {name}: {e}")
                self.state.set(name, Phase.FAILED, last_error=str(e))

    # ─── per-machine bring-up ─────────────────────────────────────────────────

    def bootstrap(self, name: str) -> bool:
        cfg = self.cfg
        h = self.handles[name]
        self.state.set(name, Phase.BOOTSTRAPPING)

        try:
            h.ensure_started(cfg.worker_machine)
        except StudioError as e:
            self.state.set(name, Phase.FAILED, last_error=str(e))
            return False
        if self.state.machines[name].started_at is None:
            self.state.set(name, started_at=time.time())

        out, code = h.exec(f"bash scripts/bootstrap_studio.sh {shlex.quote(cfg.commit)}")
        if code == 2:
            # Data missing: a duplicate that did not carry everything. Recoverable.
            print(f"▶ {name}: data incomplete, running setup_data.sh")
            out2, code2 = h.exec("bash scripts/setup_data.sh")
            if code2 != 0:
                self.state.set(name, Phase.FAILED, last_error=f"setup_data failed:\n{out2[-800:]}")
                return False
            out, code = h.exec(f"bash scripts/bootstrap_studio.sh {shlex.quote(cfg.commit)}")
        if code == 3:
            # rclone needs an interactive OAuth flow; nothing we can do remotely.
            self.state.set(name, Phase.FAILED,
                           last_error="rclone not configured on this clone; "
                                      "copy rclone.conf and retry")
            print(f"❌ {name}: {self.state.machines[name].last_error}")
            return False
        if code != 0:
            self.state.set(name, Phase.FAILED, last_error=out[-800:])
            return False

        print(f"✅ {name} bootstrapped")
        return True

    def dispatch(self, name: str, fold: int) -> bool:
        cfg = self.cfg
        h = self.handles[name]
        cmd = (f"bash scripts/run_fold.sh {fold} "
               f"{cfg.shards_per_machine} {cfg.num_workers}")
        try:
            h.exec_detached(cmd)
        except StudioError as e:
            self.state.set(name, Phase.FAILED, last_error=str(e))
            return False
        self.state.set(name, Phase.RUNNING, fold=fold, rc=None)
        print(f"🚀 {name} -> fold {fold} ({cfg.shards_per_machine} shards x "
              f"{cfg.num_workers} workers)")
        return True

    # ─── monitoring ───────────────────────────────────────────────────────────

    def monitor(self) -> None:
        cfg = self.cfg
        print(f"\n👁  polling every {cfg.poll_seconds}s — Ctrl-C is safe, "
              f"training is detached\n")
        while True:
            for name, m in self.state.machines.items():
                if Phase(m.phase) is Phase.RUNNING:
                    self._poll_running(name)
                elif Phase(m.phase) in (Phase.PROVISIONING,):
                    self._bring_up(name)

            self._print_status_line()
            if self.state.all_settled():
                print("\n✅ all folds settled")
                return
            time.sleep(cfg.poll_seconds)

    def _bring_up(self, name: str) -> None:
        if not self.bootstrap(name):
            return
        fold = self.state.claim_fold(name)
        if fold is None:
            print(f"▶ {name}: no folds left, stopping")
            self.handles[name].stop()
            self.state.set(name, Phase.STOPPED, finished_at=time.time())
            return
        self.dispatch(name, fold)

    def _poll_running(self, name: str) -> None:
        cfg, h = self.cfg, self.handles[name]
        m = self.state.machines[name]
        sentinel = f"{cfg.status_dir}/fold{m.fold}.done"

        raw = h.read_file(sentinel)
        if raw is None:
            tail = h.tail(f"{cfg.repo_dir.rsplit('/', 1)[0]}/fleet/fold{m.fold}.log")
            if tail:
                self.state.set(name, log_tail=tail[-1500:])
            return

        try:
            rc = int(raw.splitlines()[-1].strip())
        except (ValueError, IndexError):
            rc = 1
        print(f"\n{'✅' if rc == 0 else '⚠️ '} {name}: fold {m.fold} finished rc={rc}")

        self.state.set(name, Phase.COLLECTING, rc=rc)
        self.collect(name)

        m = self.state.machines[name]
        if rc == 0:
            m.folds_done.append(m.fold)
        elif m.attempts + 1 < cfg.max_attempts:
            # run_all.sh skips completed jobs now, so a retry only redoes the
            # failures -- minutes, not the whole fold.
            print(f"▶ {name}: retrying fold {m.fold} (attempt {m.attempts + 2})")
            self.state.set(name, attempts=m.attempts + 1)
            self.dispatch(name, m.fold)
            return
        else:
            print(f"⚠️  {name}: fold {m.fold} failed {cfg.max_attempts}x, moving on")

        next_fold = self.state.claim_fold(name)
        if next_fold is None:
            h.stop()
            self.state.set(name, Phase.STOPPED, finished_at=time.time(), fold=None)
            print(f"🛑 {name} stopped ({m.elapsed_hours():.2f} machine-hours)")
        else:
            self.state.set(name, attempts=0)
            self.dispatch(name, next_fold)

    def collect(self, name: str) -> None:
        """Pull reports and logs down before the machine is stopped."""
        cfg, h = self.cfg, self.handles[name]
        dest = cfg.collect_dir / name
        for remote, sub in ((f"{cfg.repo_dir}/reports", "reports"),
                            (f"{cfg.repo_dir}/logs_v4", "logs_v4")):
            try:
                h.pull_folder(remote, dest / sub)
            except Exception as e:
                print(f"⚠️  {name}: could not pull {sub}: {e} "
                      f"(results are also on Drive)")

    def _print_status_line(self) -> None:
        bits = []
        for name, m in self.state.machines.items():
            fold = f"f{m.fold}" if m.fold else "--"
            bits.append(f"{name}:{Phase(m.phase).value[:4]}/{fold}")
        print(f"  [{time.strftime('%H:%M:%S')}] " + "  ".join(bits) +
              f"   | pending={self.state.pending}"
              f" | {self.state.machine_hours():.2f}h"
              f" ≈ {self.state.credits():.1f} credits", flush=True)

    # ─── teardown / results ───────────────────────────────────────────────────

    def shutdown_all(self) -> None:
        for name, h in self.handles.items():
            if h.exists() and "running" in h.status().lower():
                print(f"🛑 stopping {name}")
                h.stop()
                self.state.set(name, Phase.STOPPED, finished_at=time.time())
        src = StudioHandle(self.cfg, self.cfg.source_studio)
        if src.exists() and "running" in src.status().lower():
            print(f"🛑 stopping {self.cfg.source_studio}")
            src.stop()

    def finalize(self) -> None:
        """Aggregate across folds locally. src/analysis.py is stdlib-only."""
        print("\n▶ aggregating locally (no credits)\n")
        for model in ("ICI", "MIR"):
            for genre in ("art", "fashion", "scenery"):
                subprocess.run(
                    ["uv", "run", "python", "-m", "src.analysis", "aggregate",
                     "--version", "v4", "--genre", genre,
                     "--pattern", "finetune", "--method", model],
                    cwd=Path(__file__).resolve().parent.parent, check=False)
        print(f"\n💳 {self.state.machine_hours():.2f} machine-hours "
              f"≈ {self.state.credits():.1f} credits "
              f"(L4 at {CREDITS_PER_HOUR}/h)")
