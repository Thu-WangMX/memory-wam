"""Exact pixels, duplicate terminal frames, and failed-seek recovery."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from fastwam.datasets.robomme import episode_video


@pytest.fixture
def videos(tmp_path):
    paths = []
    for camera in range(2):
        path = tmp_path/f'{camera}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64,48))
        assert writer.isOpened()
        rng = np.random.default_rng(camera)
        for frame in range(100):
            image = rng.integers(0, 256, (48,64,3), dtype=np.uint8)
            image[:10, :10] = frame
            writer.write(image)
        writer.release()
        paths.append(path)
    return paths


@pytest.mark.parametrize('indices', [[0,4,8,12,16], [37,41,45,49,53], [83,87,91,95,99], [99,99,99,99,99], [53,37,41,37]])
def test_seek_pixels_equal_sequential(videos, indices):
    torch.set_num_threads(1)
    with patch('fastwam.datasets.robomme.episode_paths', return_value=(None, videos)):
        old = episode_video(None, None, {'episode_index': 0}, indices, seek=False)
        new = episode_video(None, None, {'episode_index': 0}, indices)
    assert torch.equal(new, old)


def test_failed_seek_reopens_both_cameras(videos):
    real_capture = cv2.VideoCapture

    class RefuseSeek:
        def __init__(self, path):
            self.reader = real_capture(path)

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def set(self, prop, value):
            self.reader.set(prop, value)
            return False

    with patch('fastwam.datasets.robomme.episode_paths', return_value=(None, videos)):
        old = episode_video(None, None, {'episode_index': 0}, [80,84,88,92,96], seek=False)
        with patch('fastwam.datasets.robomme.cv2.VideoCapture', RefuseSeek):
            new = episode_video(None, None, {'episode_index': 0}, [80,84,88,92,96])
    assert torch.equal(new, old)


@pytest.mark.parametrize('indices', [[], [-1], [100]])
def test_invalid_or_missing_frames_fail(videos, indices):
    with patch('fastwam.datasets.robomme.episode_paths', return_value=(None, videos)):
        with pytest.raises(ValueError):
            episode_video(None, None, {'episode_index': 0}, indices)
