import math

import pytest

from src.propagation import calculate_path_loss


def test_reference_result():
    loss = calculate_path_loss(50)

    assert math.isclose(
        loss, 133.066995,
        abs_tol=0.01
    )


def test_positive_loss():
    for distance in [10, 20, 50, 100]:
        loss = calculate_path_loss(distance)

        assert math.isfinite(loss)
        assert loss > 0


def test_negative_distance():
    with pytest.raises(ValueError):
        calculate_path_loss(-10)


def test_invalid_distance():
    with pytest.raises(ValueError):
        calculate_path_loss(float("nan"))