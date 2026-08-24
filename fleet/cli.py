"""python -m fleet.cli <command>

    plan      show what a run would do; touches nothing
    up        prepare source, duplicate, bootstrap, dispatch, monitor
    status    read the state file (and optionally the live studios)
    logs      tail one machine's fold log
    collect   pull reports/logs down without stopping anything
    down      stop every studio in the fleet

`up` is resumable: state is written through on every transition and training runs
detached, so interrupting it leaves the work running. Re-run `up` to re-attach.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import CREDITS_PER_HOUR, FleetConfig
from .orchestrator import Orchestrator
from .state import FleetState, Phase


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _head_sha() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_repo_root(),
                         capture_output=True, text=True)
    return out.stdout.strip()


def _pushed(sha: str) -> bool:
    """Is this commit on the remote the studios clone from?"""
    out = subprocess.run(["git", "branch", "-r", "--contains", sha],
                         cwd=_repo_root(), capture_output=True, text=True)
    return bool(out.stdout.strip())


def build_config(args) -> FleetConfig:
    commit = args.commit or _head_sha()
    if not args.no_verify_push and not _pushed(commit):
        raise SystemExit(
            f"commit {commit[:8]} is not on any remote branch.\n"
            f"The studios clone from the remote, so they would silently run\n"
            f"stale code. Push first:  git push xun HEAD\n"
            f"(or pass --no-verify-push if you know better)"
        )
    return FleetConfig(
        commit=commit,
        machines=args.machines,
        folds=tuple(args.folds),
        shards_per_machine=args.shards,
        num_workers=args.workers,
        poll_seconds=args.poll,
        max_runtime=args.max_runtime,
    )


def cmd_plan(cfg: FleetConfig) -> int:
    per_machine = (len(cfg.folds) + cfg.machines - 1) // cfg.machines
    jobs = len(cfg.folds) * 6
    print(f"commit      {cfg.commit}")
    print(f"teamspace   {cfg.teamspace}")
    print(f"source      {cfg.source_studio} on {cfg.source_machine} "
          f"(golden image, not used for training)")
    print(f"clones      {', '.join(cfg.clone_names())} on {cfg.worker_machine}")
    print(f"folds       {list(cfg.folds)}  ->  {jobs} jobs total")
    print(f"per machine {cfg.shards_per_machine} concurrent shards x "
          f"{cfg.num_workers} workers, ~{per_machine} fold(s)")
    print(f"max_runtime {cfg.max_runtime or 'none (control is from this Mac only)'}")
    print()
    print("Estimated 15-25 machine-hours total "
          f"≈ {15 * CREDITS_PER_HOUR:.0f}-{25 * CREDITS_PER_HOUR:.0f} credits.")
    if cfg.max_runtime is None:
        print("⚠️  No max_runtime: if this process dies the studios keep billing "
              "until `fleet.cli down`.")
    return 0


def cmd_up(cfg: FleetConfig, args) -> int:
    orch = Orchestrator(cfg)
    try:
        if not args.skip_source:
            orch.prepare_source()
        orch.provision()
        orch.monitor()
        orch.finalize()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Training continues on the studios (detached).")
        print("   Re-attach with:  python -m fleet.cli up --skip-source")
        print("   Stop everything: python -m fleet.cli down")
        return 130
    return 0


def cmd_status(cfg: FleetConfig, args) -> int:
    if not cfg.state_path.exists():
        print(f"no state at {cfg.state_path}; nothing running from this machine")
        return 1
    st = FleetState.load_or_init(cfg)
    print(f"commit {st.commit[:8]}   pending folds {st.pending}")
    print(f"{'machine':<12} {'phase':<14} {'fold':<6} {'rc':<4} {'hours':<7} done")
    for name, m in st.machines.items():
        print(f"{name:<12} {Phase(m.phase).value:<14} "
              f"{str(m.fold or '-'):<6} {str(m.rc if m.rc is not None else '-'):<4} "
              f"{m.elapsed_hours():<7.2f} {m.folds_done}")
        if m.last_error:
            print(f"             ⚠️  {m.last_error.splitlines()[-1][:100]}")
    print(f"\n{st.machine_hours():.2f} machine-hours ≈ {st.credits():.1f} credits")
    if args.tail:
        for name, m in st.machines.items():
            if m.log_tail:
                print(f"\n─── {name} ───\n{m.log_tail}")
    return 0


def cmd_logs(cfg: FleetConfig, args) -> int:
    orch = Orchestrator(cfg)
    h = orch.handles.get(args.machine)
    if h is None:
        raise SystemExit(f"unknown machine {args.machine!r}; "
                         f"expected one of {cfg.clone_names()}")
    fold = orch.state.machines[args.machine].fold
    home = cfg.repo_dir.rsplit("/", 1)[0]
    print(h.tail(f"{home}/fleet/fold{fold}.log", lines=args.lines))
    return 0


def cmd_collect(cfg: FleetConfig, args) -> int:
    orch = Orchestrator(cfg)
    for name in cfg.clone_names():
        if orch.handles[name].exists():
            print(f"▶ collecting {name}")
            orch.collect(name)
    print(f"→ {cfg.collect_dir}")
    return 0


def cmd_down(cfg: FleetConfig, args) -> int:
    Orchestrator(cfg).shutdown_all()
    return 0


def main(argv=None) -> int:
    # Shared options are attached to the top level AND to every subcommand, so
    # they work on either side of it -- `fleet.cli plan --machines 1` reads more
    # naturally than argparse's default of demanding they come first.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--commit", default=None, help="SHA to pin (default: HEAD)")
    common.add_argument("--machines", type=int, default=5)
    common.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    common.add_argument("--shards", type=int, default=2,
                        help="concurrent jobs per machine")
    common.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers per shard")
    common.add_argument("--poll", type=int, default=60)
    common.add_argument("--max-runtime", default=None,
                        help="hard ceiling per studio, e.g. 10h (default: none)")
    common.add_argument("--no-verify-push", action="store_true",
                        help="skip the check that the pinned commit is on a remote")

    p = argparse.ArgumentParser(prog="fleet.cli", description=__doc__,
                                parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", parents=[common])
    up = sub.add_parser("up", parents=[common])
    up.add_argument("--skip-source", action="store_true",
                    help="do not re-prepare the golden studio (use when re-attaching)")
    stt = sub.add_parser("status", parents=[common])
    stt.add_argument("--tail", action="store_true", help="include cached log tails")
    lg = sub.add_parser("logs", parents=[common])
    lg.add_argument("machine")
    lg.add_argument("--lines", type=int, default=60)
    sub.add_parser("collect", parents=[common])
    sub.add_parser("down", parents=[common])

    args = p.parse_args(argv)
    cfg = build_config(args)

    return {
        "plan": lambda: cmd_plan(cfg),
        "up": lambda: cmd_up(cfg, args),
        "status": lambda: cmd_status(cfg, args),
        "logs": lambda: cmd_logs(cfg, args),
        "collect": lambda: cmd_collect(cfg, args),
        "down": lambda: cmd_down(cfg, args),
    }[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
