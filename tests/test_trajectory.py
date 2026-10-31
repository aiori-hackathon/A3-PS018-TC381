import math
import pytest

from src.trajectory import generate_trajectory


def test_number_of_positions():
    trajectory = generate_trajectory(0, 100, 10)

    assert len(trajectory) == 11


def test_start_and_end_positions():
    trajectory = generate_trajectory(0, 100, 10)

    assert trajectory[0]["ground_distance_km"] == 0
    assert trajectory[-1]["ground_distance_km"] == 100


def test_slant_range_increases():
    trajectory = generate_trajectory(0, 100, 10)

    slant_ranges = [
        point["slant_range_km"]
        for point in trajectory
    ]

    assert all(
        slant_ranges[i] < slant_ranges[i + 1]
        for i in range(len(slant_ranges) - 1)
    )


def test_elevation_decreases():
    trajectory = generate_trajectory(0, 100, 10)

    elevations = [
        point["elevation_deg"]
        for point in trajectory
    ]

    assert all(
        elevations[i] > elevations[i + 1]
        for i in range(len(elevations) - 1)
    )


def test_invalid_step():
    with pytest.raises(ValueError):
        generate_trajectory(0, 100, 0)


def test_invalid_distance():
    with pytest.raises(ValueError):
        generate_trajectory(-10, 100, 10)


def test_non_divisible_endpoint():
    trajectory = generate_trajectory(0, 25, 10)

    distances = [
        point["ground_distance_km"]
        for point in trajectory
    ]

    assert distances == [0.0, 10.0, 20.0]


def test_geometry_consistency():
    trajectory = generate_trajectory(0, 100, 10)

    assert math.isclose(
        trajectory[0]["slant_range_km"],
        20.0,
        abs_tol=1e-6
    )