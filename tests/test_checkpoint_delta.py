"""Equivalence tests for trainable-only finetune checkpoints.

`trainer_finetune` saves only the part of the model that finetuning can change
and restores the frozen NIMA weights from the pretrain checkpoint at load time.
These tests pin the invariants that make that substitution weight-for-weight
identical to saving the whole state dict.
"""
import argparse
import copy

import pytest
import torch

from src.train_common import (FROZEN_STATE_PREFIX, build_piaa_model,
                              is_trainable_only_state, num_bins, trainable_state)

NUM_ATTR, NUM_PT = 45, 116
GENRES = ['art']
BACKBONE_DICT = {'art': 'clip_vit_b16'}


def _build(model_type):
    args = argparse.Namespace(model_type=model_type, dropout=0.1)
    return build_piaa_model(num_bins, NUM_ATTR, NUM_PT, GENRES, BACKBONE_DICT, args)


def _assert_same_state(a, b):
    assert set(a) == set(b), f"key mismatch: {sorted(set(a) ^ set(b))[:5]}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor differs: {k}"


@pytest.fixture(scope='module', params=['ICI', 'MIR'])
def model(request):
    return _build(request.param), request.param


def test_split_is_exhaustive_and_disjoint(model):
    m, _ = model
    full = m.state_dict()
    kept = set(trainable_state(full))
    dropped = {k for k in full if k.startswith(FROZEN_STATE_PREFIX)}
    assert kept | dropped == set(full)
    assert not kept & dropped


def test_freeze_backbone_matches_the_prefix(model):
    """Everything under nima_dict is frozen, and nothing else is."""
    m, _ = model
    m.freeze_backbone()
    for name, param in m.named_parameters():
        assert param.requires_grad != name.startswith(FROZEN_STATE_PREFIX), name


def test_optimizer_cannot_reach_a_dropped_tensor(model):
    m, _ = model
    m.freeze_backbone()
    optimized = {id(p) for p in filter(lambda p: p.requires_grad, m.parameters())}
    for name, param in m.named_parameters():
        if name.startswith(FROZEN_STATE_PREFIX):
            assert id(param) not in optimized, name


def test_pretrain_plus_delta_reproduces_the_full_state(model):
    m, model_type = model
    m.freeze_backbone()
    base = copy.deepcopy(m.state_dict())

    with torch.no_grad():  # stand in for a finetuning run
        for param in m.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.01)
    trained = copy.deepcopy(m.state_dict())

    delta = trainable_state(trained)
    assert is_trainable_only_state(delta)
    assert not is_trainable_only_state(trained)

    restored = _build(model_type)
    restored.load_state_dict(base, strict=False)
    restored.load_state_dict(delta, strict=False)
    _assert_same_state(restored.state_dict(), trained)


def test_reused_model_matches_a_rebuilt_one(model):
    """Reloading the pretrain state resets a reused model as fully as a rebuild."""
    m, model_type = model
    base = copy.deepcopy(m.state_dict())

    with torch.no_grad():
        for param in m.parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.01)

    m.load_state_dict(base, strict=False)
    rebuilt = _build(model_type)
    incompatible = rebuilt.load_state_dict(base, strict=False)
    assert not incompatible.missing_keys
    _assert_same_state(m.state_dict(), rebuilt.state_dict())
