import numpy as np
import pytest

from examples.crater_detect_data import (
    CELL_SIZE,
    GRID_SIZE,
    RADIUS_SCALE,
    TILE_SIZE,
    CraterParams,
    _max_center_error,
    _radius_error,
    assign_targets,
    decode_target,
    make_detection_dataset,
    make_synthetic_image,
)


def test_empty_target_is_all_zeros():
    target, info = assign_targets([])
    assert info.total_craters == 0
    assert info.assigned == 0
    assert info.conflicts == 0
    assert np.all(target == 0.0)


def test_single_crater_center_target():
    crater = CraterParams(cx=32.0, cy=32.0, radius=8.0)
    target, info = assign_targets([crater])
    assert info.assigned == 1
    assert info.conflicts == 0

    gx = int(32.0 // CELL_SIZE)
    gy = int(32.0 // CELL_SIZE)
    assert target[gy, gx, 0] == 1.0
    assert target[gy, gx, 1] == pytest.approx(0.0, abs=1e-6)
    assert target[gy, gx, 2] == pytest.approx(0.0, abs=1e-6)
    expected_log_r = np.log(8.0 / RADIUS_SCALE)
    assert target[gy, gx, 3] == pytest.approx(expected_log_r, rel=1e-6)

    non_assigned = np.sum(target[:, :, 0] > 0.5)
    assert non_assigned == 1


def test_single_crater_offset_target():
    crater = CraterParams(cx=10.0, cy=22.0, radius=5.0)
    target, info = assign_targets([crater])
    assert info.assigned == 1

    gx = 1
    gy = 2
    expected_dx = (10.0 - 1 * 8) / 8.0
    expected_dy = (22.0 - 2 * 8) / 8.0
    assert target[gy, gx, 0] == 1.0
    assert target[gy, gx, 1] == pytest.approx(expected_dx, rel=1e-6)
    assert target[gy, gx, 2] == pytest.approx(expected_dy, rel=1e-6)


def test_decode_inverts_assign_for_various_positions():
    rng = np.random.RandomState(42)
    craters = [
        CraterParams(
            cx=rng.uniform(4.0, TILE_SIZE - 4.0),
            cy=rng.uniform(4.0, TILE_SIZE - 4.0),
            radius=rng.uniform(3.0, 12.0),
        )
        for _ in range(4)
    ]
    target, info = assign_targets(craters)
    assert info.conflicts == 0
    assert info.assigned == len(craters)

    decoded = decode_target(target)
    assert len(decoded) == len(craters)
    assert _max_center_error(craters, decoded) < 1.0
    assert _radius_error(craters, decoded) < 0.1


def test_conflict_recorded_when_two_craters_in_same_cell():
    craters = [
        CraterParams(cx=4.0, cy=4.0, radius=5.0),
        CraterParams(cx=6.0, cy=6.0, radius=5.0),
    ]
    target, info = assign_targets(craters)
    assert info.total_craters == 2
    assert info.assigned == 1
    assert info.conflicts == 1
    assert np.sum(target[:, :, 0] > 0.5) == 1


def test_out_of_bounds_recorded():
    craters = [
        CraterParams(cx=-1.0, cy=32.0, radius=5.0),
        CraterParams(cx=65.0, cy=32.0, radius=5.0),
    ]
    _, info = assign_targets(craters)
    assert info.out_of_bounds == 2
    assert info.assigned == 0


def test_synthetic_image_produces_valid_range():
    craters = [CraterParams(32.0, 32.0, 8.0)]
    img = make_synthetic_image(craters, seed=42)
    assert img.shape == (1, TILE_SIZE, TILE_SIZE)
    assert img.dtype == np.float32
    assert -1.0 <= img.min() <= 1.0
    assert -1.0 <= img.max() <= 1.0


def test_dataset_is_deterministic():
    imgs1, tgts1, infos1 = make_detection_dataset(count=4, seed=42)
    imgs2, tgts2, infos2 = make_detection_dataset(count=4, seed=42)
    np.testing.assert_array_equal(imgs1, imgs2)
    np.testing.assert_array_equal(tgts1, tgts2)

    imgs3, tgts3, _ = make_detection_dataset(count=4, seed=99)
    assert not np.array_equal(tgts1, tgts3)


def test_dataset_shapes():
    imgs, tgts, infos = make_detection_dataset(count=3, seed=7)
    assert imgs.shape == (3, 1, TILE_SIZE, TILE_SIZE)
    assert tgts.shape == (3, GRID_SIZE, GRID_SIZE, 4)
    assert len(infos) == 3
    for info in infos:
        assert info.total_craters >= 1
        assert info.assigned >= 1


def test_empty_grid_cells_have_zero_objectness():
    craters = [CraterParams(4.0, 4.0, 5.0)]
    target, _ = assign_targets(craters)
    total_assigned = np.sum(target[:, :, 0] > 0.5)
    assert total_assigned == 1
    total_zero = np.sum(target[:, :, 0] == 0.0)
    assert total_zero == GRID_SIZE * GRID_SIZE - 1
