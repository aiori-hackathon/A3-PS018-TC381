import math

from src.geometry import calculate_geometry


def test_receiver_directly_below_haps():
    slant, elevation = calculate_geometry(0)

    assert math.isclose(slant, 20.0, abs_tol=1e-6)
    assert math.isclose(elevation, 90.0, abs_tol=1e-6)


def test_slant_range_increases():
    slant_15, _ = calculate_geometry(15)
    slant_50, _ = calculate_geometry(50)
    slant_100, _ = calculate_geometry(100)

    assert slant_15 < slant_50 < slant_100


def test_negative_distance_rejected():
    try:
        calculate_geometry(-10)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative distance should raise ValueError"
        )