"""The shard partition run_all.sh uses to fan out concurrent jobs.

Two shards on one machine only stay safe while the partition is exhaustive and
disjoint. The subtle failure is drift: if the round-robin counter advanced only
for jobs that actually run, a retry -- where some jobs are skipped as complete --
would re-partition, and two shards could land on the same (model, genre), both
writing the same report prefix and the same models_pth directory.

These tests drive the real script through `--dry-run`, which short-circuits the
dataset check, so they need no data, no GPU and no torch.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / 'run_all.sh'
JOB_RE = re.compile(r'\[Dry Run\] Finetune: (\S+) \| (\S+) \| (\S+)')
SKIP_RE = re.compile(r'Skipping already completed: (\S+) \| (\S+) \| (\S+)')


def _run(*args):
    proc = subprocess.run(
        ['bash', str(SCRIPT), '--dry-run', '--folds', '1', *args],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"run_all.sh failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def planned(*args):
    """(model, genre) pairs the invocation would actually run."""
    return {(m, g) for _, m, g in JOB_RE.findall(_run(*args))}


def skipped(*args):
    return {(m, g) for _, m, g in SKIP_RE.findall(_run(*args))}


@pytest.fixture
def completed(request):
    """Mark (genre, model) jobs done by planting the report JSON they emit."""
    written = []

    def mark(pairs):
        for model, genre in pairs:
            d = REPO / 'reports' / 'exp' / 'v4_fold1' / genre
            d.mkdir(parents=True, exist_ok=True)
            f = d / f'{genre}_{model}_x_finetune.json'
            f.write_text('{}')
            written.append(f)

    yield mark
    for f in written:
        f.unlink(missing_ok=True)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_shards_partition_the_work(n):
    whole = planned()
    assert len(whole) == 6

    shards = [planned('--shard', f'{i}/{n}') for i in range(1, n + 1)]
    union = set().union(*shards)
    assert union == whole, "some jobs would never run"
    assert sum(len(s) for s in shards) == len(whole), "a job is claimed by two shards"


@pytest.mark.parametrize('n', [2, 3])
def test_shards_are_balanced(n):
    sizes = [len(planned('--shard', f'{i}/{n}')) for i in range(1, n + 1)]
    assert max(sizes) - min(sizes) <= 1, f"lopsided split: {sizes}"


def test_two_shards_each_get_one_of_every_genre():
    """Round-robin over model x genre; a per-model split would not do this."""
    for i in (1, 2):
        genres = sorted(g for _, g in planned('--shard', f'{i}/2'))
        assert genres == ['art', 'fashion', 'scenery']


def test_assignment_does_not_drift_when_jobs_are_already_done(completed):
    """A retry must keep every shard on exactly the jobs it owned before."""
    before = {i: planned('--shard', f'{i}/2') for i in (1, 2)}

    completed([('ICI', 'art'), ('MIR', 'scenery')])

    for i in (1, 2):
        still_planned = planned('--shard', f'{i}/2')
        was_skipped = skipped('--shard', f'{i}/2')
        assert still_planned | was_skipped == before[i], (
            f"shard {i} changed hands on retry")
    assert planned('--shard', '1/2') & planned('--shard', '2/2') == set()


def test_completed_jobs_are_skipped_and_force_overrides(completed):
    completed([('ICI', 'art')])
    assert ('ICI', 'art') not in planned()
    assert ('ICI', 'art') in skipped()
    assert ('ICI', 'art') in planned('--force')


def test_jobs_flag_fans_out_the_same_partition():
    out = _run('--jobs', '2')
    assert out.count('────────── shard') == 2
    assert {(m, g) for _, m, g in JOB_RE.findall(out)} == planned()


def test_partition_holds_for_a_single_model():
    whole = planned('--models', 'ICI')
    assert len(whole) == 3
    shards = [planned('--models', 'ICI', '--shard', f'{i}/2') for i in (1, 2)]
    assert set().union(*shards) == whole
    assert sum(len(s) for s in shards) == len(whole)
