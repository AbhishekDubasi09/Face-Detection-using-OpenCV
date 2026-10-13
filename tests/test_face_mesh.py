"""Smoke test: the model loads and finds a full face mesh on the bundled sample image."""
import os
import sys

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    img = cv2.imread(main.DEFAULT_IMAGE)
    assert img is not None, "bundled sample image is missing"
    return img


def test_finds_478_landmarks_inside_the_image(sample):
    landmarks = main.get_landmarks(sample)
    assert landmarks is not None
    assert len(landmarks) == 478
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    assert 0.0 <= min(xs) and max(xs) <= 1.0
    assert 0.0 <= min(ys) and max(ys) <= 1.0


def test_no_face_returns_none():
    blank = cv2.cvtColor(cv2.UMat(240, 320, cv2.CV_8UC1, 128).get(), cv2.COLOR_GRAY2BGR)
    assert main.get_landmarks(blank) is None


def test_draw_mesh_changes_the_image(sample):
    img = sample.copy()
    main.draw_mesh(img, main.get_landmarks(sample))
    assert (img != sample).any()
