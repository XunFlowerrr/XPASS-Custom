"""Fleet state: the part that has to survive the orchestrator being killed.

Training runs detached on the studios, so the orchestrator is only a monitor --
but it holds the record of which fold went where and how many machine-hours have
been billed. Losing or corrupting that means re-running folds that already
finished, on machines that are still charging.

None of this needs lightning-sdk; `fleet.state` and `fleet.config` are pure.
"""
import json

import pytest

from fleet.config import CREDITS_PER_HOUR, FleetConfig
from fleet.state import FleetState, Phase


@pytest.fixture
def cfg(tmp_path):
    return FleetConfig(commit="deadbeef", machines=3, folds=(1, 2, 3, 4, 5),
                       state_path=tmp_path / "fleet_state.json")


def test_starts_with_every_fold_pending(cfg):
    st = FleetState.load_or_init(cfg)
    assert st.pending == [1, 2, 3, 4, 5]
    assert sorted(st.machines) == ["xpass-f1", "xpass-f2", "xpass-f3"]
    assert all(Phase(m.phase) is Phase.PLANNED for m in st.machines.values())


def test_a_fold_is_claimed_once(cfg):
    st = FleetState.load_or_init(cfg)
    claims = [st.claim_fold(n) for n in ("xpass-f1", "xpass-f2", "xpass-f3")]
    assert claims == [1, 2, 3]
    assert st.pending == [4, 5]
    assert len(set(claims)) == 3


def test_more_folds_than_machines_drain_through_work_stealing(cfg):
    """5 folds over 3 machines: whoever frees up takes the next one."""
    st = FleetState.load_or_init(cfg)
    for n in st.machines:
        st.claim_fold(n)
    assert st.claim_fold("xpass-f1") == 4      # f1 finished first
    assert st.claim_fold("xpass-f1") == 5      # and again
    assert st.claim_fold("xpass-f2") is None   # nothing left
    assert st.pending == []


def test_released_fold_goes_back_to_the_front(cfg):
    """A machine that dies must not take its fold down with it."""
    st = FleetState.load_or_init(cfg)
    st.claim_fold("xpass-f1")
    st.claim_fold("xpass-f2")
    st.release_fold("xpass-f1")
    assert st.pending[0] == 1
    assert st.machines["xpass-f1"].fold is None
    assert st.claim_fold("xpass-f3") == 1


def test_release_is_idempotent(cfg):
    st = FleetState.load_or_init(cfg)
    st.claim_fold("xpass-f1")
    st.release_fold("xpass-f1")
    st.release_fold("xpass-f1")
    assert st.pending.count(1) == 1


def test_state_round_trips_through_disk(cfg):
    st = FleetState.load_or_init(cfg)
    st.claim_fold("xpass-f1")
    st.set("xpass-f1", Phase.RUNNING, started_at=1000.0, attempts=1)

    again = FleetState.load_or_init(cfg)
    m = again.machines["xpass-f1"]
    assert Phase(m.phase) is Phase.RUNNING
    assert (m.fold, m.attempts, m.started_at) == (1, 1, 1000.0)
    assert again.pending == [2, 3, 4, 5]


def test_every_transition_is_persisted(cfg):
    """No explicit save() call should ever be required."""
    st = FleetState.load_or_init(cfg)
    st.set("xpass-f2", Phase.BOOTSTRAPPING)
    on_disk = json.loads(cfg.state_path.read_text())
    assert on_disk["machines"]["xpass-f2"]["phase"] == "bootstrapping"


def test_save_never_leaves_a_partial_file(cfg):
    st = FleetState.load_or_init(cfg)
    for i in range(20):
        st.set("xpass-f1", Phase.RUNNING, attempts=i)
        json.loads(cfg.state_path.read_text())   # would raise if half-written
    assert not cfg.state_path.with_suffix(".json.tmp").exists()


def test_a_state_file_from_another_commit_is_refused(cfg):
    FleetState.load_or_init(cfg)
    other = FleetConfig(commit="cafe1234", machines=3,
                        state_path=cfg.state_path)
    with pytest.raises(SystemExit, match="deadbeef"):
        FleetState.load_or_init(other)


def test_billing_is_reported_in_credits(cfg):
    st = FleetState.load_or_init(cfg)
    st.set("xpass-f1", started_at=0.0, finished_at=3600.0)     # 1 h
    st.set("xpass-f2", started_at=0.0, finished_at=1800.0)     # 0.5 h
    assert st.machine_hours() == pytest.approx(1.5)
    assert st.credits() == pytest.approx(1.5 * CREDITS_PER_HOUR)


def test_settled_needs_both_an_empty_queue_and_idle_machines(cfg):
    st = FleetState.load_or_init(cfg)
    st.pending.clear()
    st.set("xpass-f1", Phase.RUNNING)
    st.set("xpass-f2", Phase.STOPPED)
    st.set("xpass-f3", Phase.DONE)
    assert not st.all_settled(), "a running machine is not settled"

    st.set("xpass-f1", Phase.STOPPED)
    assert st.all_settled()

    st.pending.append(4)
    assert not st.all_settled(), "queued work is not settled"


def test_remote_env_is_absolute_and_sets_path():
    """$HOME/bin holds rclone; nothing on the studio guarantees it is on PATH."""
    cfg = FleetConfig(commit="x")
    env = cfg.remote_env()
    assert "$HOME/bin" in env
    assert cfg.repo_dir.startswith("/teamspace/")
    assert "cd /teamspace/" in env
