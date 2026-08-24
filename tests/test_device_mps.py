import pytest
import torch
import torch.nn.functional as F

from src.argflags import get_device, parse_arguments
from src.train_common import NIMA, PIAA_ICI, PIAA_MIR, num_bins, earth_mover_distance
from src.methods.source_only import _get_autocast, _get_grad_scaler


def test_get_device():
    # Test explicit CPU
    dev_cpu = get_device('cpu')
    assert dev_cpu.type == 'cpu'

    # Test auto detection
    dev_auto = get_device('auto')
    assert isinstance(dev_auto, torch.device)

    # Test MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        assert dev_auto.type == 'mps'
        dev_mps = get_device('mps')
        assert dev_mps.type == 'mps'


def test_autocast_and_scaler():
    device = get_device('auto')
    autocast_ctx = _get_autocast(device)
    scaler = _get_grad_scaler(device)

    assert autocast_ctx is not None
    assert scaler is not None

    x = torch.randn(4, 4, requires_grad=True, device=device)
    with autocast_ctx:
        y = x.pow(2).sum()

    scaler.scale(y).backward()
    assert x.grad is not None


def test_nima_forward_backward_on_device():
    device = get_device('auto')
    model = NIMA(num_bins, backbone='clip_vit_b16').to(device)
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = _get_grad_scaler(device)
    autocast_ctx = _get_autocast(device)

    # Batch of images [B, 3, 224, 224]
    dummy_images = torch.randn(2, 3, 224, 224, device=device)
    dummy_hist = F.softmax(torch.randn(2, num_bins, device=device), dim=1)

    optimizer.zero_grad()
    with autocast_ctx:
        logits = model(dummy_images)
        prob = F.softmax(logits, dim=1)
        loss = earth_mover_distance(prob, dummy_hist).mean()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not torch.isnan(loss)


def test_piaa_ici_forward_backward_on_device():
    device = get_device('auto')
    genres = ['art']
    backbone_dict = {'art': 'clip_vit_b16'}
    num_attr = 10
    num_pt = 10

    class DummyArgs:
        dropout = 0.1

    model = PIAA_ICI(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=0.1).to(device)
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = _get_grad_scaler(device)
    autocast_ctx = _get_autocast(device)

    dummy_images = torch.randn(2, 3, 224, 224, device=device)
    dummy_pt = torch.randn(2, num_pt, device=device)
    dummy_attr = torch.randn(2, num_attr, device=device)
    dummy_target = torch.randn(2, 1, device=device)

    optimizer.zero_grad()
    with autocast_ctx:
        pred = model(dummy_images, dummy_pt, dummy_attr, 'art')
        loss = F.mse_loss(pred, dummy_target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not torch.isnan(loss)


def test_piaa_mir_forward_backward_on_device():
    device = get_device('auto')
    genres = ['art']
    backbone_dict = {'art': 'clip_vit_b16'}
    num_attr = 10
    num_pt = 10

    model = PIAA_MIR(num_bins, num_attr, num_pt, genres, backbone_dict, dropout=0.1).to(device)
    model.freeze_backbone()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = _get_grad_scaler(device)
    autocast_ctx = _get_autocast(device)

    dummy_images = torch.randn(2, 3, 224, 224, device=device)
    dummy_pt = torch.randn(2, num_pt, device=device)
    dummy_attr = torch.randn(2, num_attr, device=device)
    dummy_target = torch.randn(2, 1, device=device)

    optimizer.zero_grad()
    with autocast_ctx:
        pred = model(dummy_images, dummy_pt, dummy_attr, 'art')
        loss = F.mse_loss(pred, dummy_target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not torch.isnan(loss)


def test_torch_load_map_location(tmp_path):
    device = get_device('auto')
    tensor = torch.randn(5, 5)
    save_path = tmp_path / "tensor.pth"
    torch.save(tensor, save_path)

    loaded = torch.load(save_path, map_location=device)
    assert loaded.device.type == device.type
