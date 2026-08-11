"""Colour tracking, target locking, and error normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import draw_bgr_box, make_frame
from obj_drone.vision.detector import Detection
from obj_drone.vision.tracker import TargetTracker

RED_BGR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)


@pytest.fixture
def tracker() -> TargetTracker:
    return TargetTracker(frame_width=640, frame_height=480)


def test_detects_red_blob_not_blue(tracker: TargetTracker) -> None:
    """The default HSV range tracks red. A BGR frame must be interpreted as BGR.

    This is the regression test for the picamera2 RGB888/BGR mix-up: if frames
    arrive with R and B swapped, a red target reads as blue and is never found.
    """
    red_frame = draw_bgr_box(make_frame(), RED_BGR, 300, 200, 60, 60)
    result = tracker.detect_color(red_frame)
    assert result.found
    assert result.bbox is not None

    tracker.reset_lock()
    blue_frame = draw_bgr_box(make_frame(), BLUE_BGR, 300, 200, 60, 60)
    assert not tracker.detect_color(blue_frame).found


def test_blob_center_is_box_center(tracker: TargetTracker) -> None:
    frame = draw_bgr_box(make_frame(), RED_BGR, 100, 150, 80, 40)
    result = tracker.detect_color(frame)
    assert result.found
    assert result.center_x == pytest.approx(140, abs=3)
    assert result.center_y == pytest.approx(170, abs=3)


def test_blob_below_min_area_is_ignored() -> None:
    tracker = TargetTracker(frame_width=640, frame_height=480, min_blob_area=5000)
    frame = draw_bgr_box(make_frame(), RED_BGR, 300, 200, 10, 10)
    assert not tracker.detect_color(frame).found


def test_pixel_and_normalized_error(tracker: TargetTracker) -> None:
    frame = draw_bgr_box(make_frame(), RED_BGR, 480, 360, 40, 40)
    result = tracker.detect_color(frame)
    assert result.found

    err_x, err_y = tracker.pixel_error(result)
    assert err_x == pytest.approx(180, abs=3)
    assert err_y == pytest.approx(140, abs=3)

    nx, ny = tracker.normalized_error(result)
    assert nx == pytest.approx(180 / 320, abs=0.02)
    assert ny == pytest.approx(140 / 240, abs=0.02)


def test_centered_target_has_zero_error(tracker: TargetTracker) -> None:
    frame = draw_bgr_box(make_frame(), RED_BGR, 310, 230, 20, 20)
    result = tracker.detect_color(frame)
    assert result.found
    nx, ny = tracker.normalized_error(result)
    assert nx == pytest.approx(0.0, abs=0.02)
    assert ny == pytest.approx(0.0, abs=0.02)


def _det(cx: float, cy: float, conf: float, label: str = "person") -> Detection:
    w = h = 40
    return Detection(
        class_id=0,
        label=label,
        confidence=conf,
        bbox=(int(cx - w / 2), int(cy - h / 2), w, h),
    )


def test_select_target_picks_highest_confidence_first(tracker: TargetTracker) -> None:
    result = tracker.select_target([_det(100, 100, 0.9), _det(500, 400, 0.6)])
    assert result.found
    assert result.center_x == pytest.approx(100, abs=1)
    assert result.confidence == pytest.approx(0.9)


def test_select_target_keeps_lock_on_nearest(tracker: TargetTracker) -> None:
    """Once locked, follow the nearby target even if another scores higher."""
    tracker.select_target([_det(100, 100, 0.9)])
    result = tracker.select_target([_det(110, 105, 0.5), _det(500, 400, 0.95)])
    assert result.center_x == pytest.approx(110, abs=1)


def test_lock_prefers_nearest_not_most_confident() -> None:
    """Once locked, proximity beats confidence — we follow one subject."""
    tracker = TargetTracker(640, 480, max_track_jump_px=200, reid_enabled=False)
    tracker.select_target([_det(100, 100, 0.9)])
    result = tracker.select_target([_det(140, 110, 0.4), _det(600, 400, 0.99)])
    assert result.center_x == pytest.approx(140, abs=1)


def test_lock_survives_frames_with_no_detections(tracker: TargetTracker) -> None:
    """The subject vanishing for a moment must not hand the lock to someone else.

    This is the core of "stick with the first person you see": an occlusion or a
    dropped detection cannot cause the drone to start following a stranger.
    """
    tracker.select_target([_det(100, 100, 0.9)])
    assert tracker.locked

    for _ in range(30):
        assert not tracker.select_target([]).found
    assert tracker.locked, "lock must persist through a gap in detections"

    # Our subject reappears near where they were, next to a more confident
    # stranger across the frame. We must go back to our subject.
    result = tracker.select_target([_det(115, 108, 0.3), _det(500, 400, 0.99)])
    assert result.center_x == pytest.approx(115, abs=2)


def test_first_person_wins_and_lock_never_transfers() -> None:
    tracker = TargetTracker(640, 480, reid_enabled=False, max_track_jump_px=120)
    first = tracker.select_target([_det(200, 240, 0.7)])
    assert first.center_x == pytest.approx(200, abs=1)

    # A much stronger detection appears elsewhere every frame; ignore it.
    for _ in range(20):
        r = tracker.select_target([_det(210, 240, 0.5), _det(600, 100, 0.99)])
        assert r.center_x < 400, "lock jumped to a different subject"


def test_reset_lock_allows_new_acquisition(tracker: TargetTracker) -> None:
    tracker.select_target([_det(100, 100, 0.9)])
    tracker.reset_lock()
    assert not tracker.locked
    result = tracker.select_target([_det(500, 400, 0.8)])
    assert result.center_x == pytest.approx(500, abs=1)


def test_appearance_distinguishes_differently_coloured_subjects() -> None:
    """Re-ID must tell two people apart by clothing colour."""
    import numpy as np

    frame = make_frame()
    # Subject A in red on the left, subject B in blue on the right.
    draw_bgr_box(frame, (0, 0, 255), 80, 80, 60, 120)
    draw_bgr_box(frame, (255, 0, 0), 400, 80, 60, 120)
    tracker = TargetTracker(640, 480)

    sig_a = tracker._appearance(frame, (80, 80, 60, 120))
    sig_b = tracker._appearance(frame, (400, 80, 60, 120))
    assert sig_a is not None and sig_b is not None
    assert tracker._similarity(sig_a, sig_a) > 0.99
    assert tracker._similarity(sig_a, sig_b) < 0.5


def test_appearance_handles_degenerate_boxes() -> None:
    tracker = TargetTracker(640, 480)
    assert tracker._appearance(make_frame(), (0, 0, 0, 0)) is None
    assert tracker._appearance(make_frame(), (10, 10, 2, 2)) is None
    assert tracker._similarity(None, None) == 0.0


def test_detect_objects_without_detector_raises(tracker: TargetTracker) -> None:
    with pytest.raises(RuntimeError, match="No object detector"):
        tracker.detect_objects(make_frame())


def test_red_hue_wraparound_is_covered(tracker: TargetTracker) -> None:
    """Red spans hue 0, so the mask must include the top of the hue wheel."""
    frame = draw_bgr_box(make_frame(), (0, 0, 200), 300, 200, 50, 50)
    mask = tracker.color_mask(frame)
    assert np.count_nonzero(mask) > 2000
