import math
from functools import lru_cache

from Py528.P528 import bt_loss


# Fixed simulation assumptions
FREQUENCY_MHZ = 2000.0
RECEIVER_HEIGHT_M = 1.5
HAPS_HEIGHT_M = 20000.0
TIME_PERCENTAGE = 50.0
POLARIZATION = 1  # Vertical


@lru_cache(maxsize=4096)
def calculate_path_loss(ground_distance_km):
    """
    Calculate basic transmission loss using ITU-R P.528-5.

    Input:
        ground_distance_km: Ground-path distance in km.

    Returns:
        Basic transmission loss in dB.
    """

    if not math.isfinite(ground_distance_km):
        raise ValueError("Distance must be finite.")

    if ground_distance_km < 0:
        raise ValueError("Distance cannot be negative.")

    result = bt_loss(
        d_km=ground_distance_km,
        h_1_meter=RECEIVER_HEIGHT_M,
        h_2_meter=HAPS_HEIGHT_M,
        f_mhz=FREQUENCY_MHZ,
        T_pol=POLARIZATION,
        p=TIME_PERCENTAGE,
        use_reflection=True
    )

    if result.rtn != 0:
        raise RuntimeError(
            f"P.528 calculation failed: code {result.rtn}"
        )

    return float(result.A_db)


if __name__ == "__main__":
    for distance in [10, 20, 50, 100]:
        loss = calculate_path_loss(distance)

        print(
            f"Distance: {distance:6.1f} km | "
            f"Path loss: {loss:8.3f} dB"
        )