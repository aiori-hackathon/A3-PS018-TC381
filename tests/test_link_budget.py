import math

from src.link_budget import calculate_link_budget


def test_received_power():
    result = calculate_link_budget(50)

    expected = 55 - result["path_loss_db"]

    assert math.isclose(
        result["received_power_dbm"],
        expected,
        abs_tol=1e-6
    )


def test_noise_power():
    result = calculate_link_budget(50)

    assert math.isclose(
        result["noise_power_dbm"],
        -99.0,
        abs_tol=1e-6
    )


def test_snr_calculation():
    result = calculate_link_budget(50)

    expected = (
        result["received_power_dbm"]
        - result["noise_power_dbm"]
    )

    assert math.isclose(
        result["snr_db"],
        expected,
        abs_tol=1e-6
    )


def test_snr_decreases_with_distance():
    near = calculate_link_budget(10)
    far = calculate_link_budget(100)

    assert near["snr_db"] > far["snr_db"]