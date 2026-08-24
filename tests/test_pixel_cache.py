"""The decoded-image cache must be invisible to everything downstream.

`warm_pixel_cache` exists purely to stop the same JPEGs being decoded on every
epoch. These tests pin the property that makes it safe: the transform receives
the same pixels whether or not the cache is warm, so augmentation is drawn
exactly as it is today.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data import ImageDataset, make_loader


def _bare_dataset(tmp_path, genre, files):
    """An ImageDataset with only the attributes the image path touches."""
    ds = ImageDataset.__new__(ImageDataset)
    ds.genre = genre
    ds.samples_dir = str(tmp_path)
    ds.data = pd.DataFrame({'sample_file': files})
    ds._pixel_cache = None
    return ds


@pytest.fixture
def images(tmp_path):
    d = tmp_path / 'art'
    d.mkdir()
    rng = np.random.default_rng(0)
    names = []
    for i in range(6):
        arr = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f'{i}.jpg', quality=95)
        names.append(f'{i}.jpg')
    return tmp_path, names


def test_cache_returns_identical_pixels(images):
    tmp_path, names = images
    ds = _bare_dataset(tmp_path, 'art', names)
    fresh = [np.asarray(ds._open_image(ds._resolve_sample(i)[1])) for i in range(len(names))]

    cached_bytes = ds.warm_pixel_cache(1 << 30)
    assert cached_bytes > 0
    for i, expected in enumerate(fresh):
        got = np.asarray(ds._open_image(ds._resolve_sample(i)[1]))
        assert np.array_equal(got, expected), f"pixels differ for {names[i]}"


def test_transform_output_is_bit_identical(images):
    """Same seed, same augmentation -- cache or no cache."""
    tmp_path, names = images
    tf = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                             transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                             transforms.ToTensor()])

    def draw(ds):
        torch.manual_seed(1234)
        return [tf(ds._open_image(ds._resolve_sample(i)[1])) for i in range(len(names))]

    cold = _bare_dataset(tmp_path, 'art', names)
    warm = _bare_dataset(tmp_path, 'art', names)
    warm.warm_pixel_cache(1 << 30)

    for a, b in zip(draw(cold), draw(warm)):
        assert torch.equal(a, b)


def test_budget_zero_disables_the_cache(images):
    tmp_path, names = images
    ds = _bare_dataset(tmp_path, 'art', names)
    assert ds.warm_pixel_cache(0) == 0
    assert ds._pixel_cache is None


def test_budget_caps_and_still_serves_the_rest(images):
    """Images past the budget are skipped, not corrupted."""
    tmp_path, names = images
    ds = _bare_dataset(tmp_path, 'art', names)
    one = 60 * 80 * 3
    ds.warm_pixel_cache(one * 2 + 1)
    assert 0 < len(ds._pixel_cache) < len(names)
    for i in range(len(names)):
        path = ds._resolve_sample(i)[1]
        assert np.array_equal(np.asarray(ds._open_image(path)),
                              np.asarray(Image.open(path).convert('RGB')))


def test_scenery_mp4_names_map_to_jpg(tmp_path):
    d = tmp_path / 'scenery_image'
    d.mkdir()
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(d / 'clip.jpg')
    ds = _bare_dataset(d, 'scenery', ['clip.mp4'])
    name, path = ds._resolve_sample(0)
    assert name == 'clip.jpg' and os.path.basename(path) == 'clip.jpg'


class _Tiny(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return {'image': torch.zeros(1), 'Aesthetic': torch.zeros(1),
                                      'traits': torch.zeros(1), 'QIP': torch.zeros(1)}


def test_reused_loaders_keep_workers_single_pass_ones_do_not():
    reused = make_loader(_Tiny(), 2, 2, reused=True)
    once = make_loader(_Tiny(), 2, 2, reused=False)
    assert reused.persistent_workers is True
    assert once.persistent_workers is False
    del reused, once


def test_worker_options_are_omitted_without_workers():
    """persistent_workers/prefetch_factor are invalid when num_workers == 0."""
    dl = make_loader(_Tiny(), 2, 0)
    assert dl.num_workers == 0 and dl.persistent_workers is False
