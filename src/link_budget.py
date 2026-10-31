import math

from src.propagation import calculate_path_loss


# Link-budget assumptions
TX_POWER_DBM = 30.0
TX_GAIN_DBI = 20.0
RX_GAIN_DBI = 5.0

BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 5.0
THERMAL_NOISE_DBM_HZ = -174.0


def calculate_link_budget(ground_distance_km):
    """
    Calculate received power and SNR for a HAPS link.
    """

    # Propagation loss from ITU-R P.528
    path_loss_db = calculate_path_loss(
        ground_distance_km
    )

    # Received signal power (dBm)
    received_power_dbm = (
        TX_POWER_DBM
        + TX_GAIN_DBI
        + RX_GAIN_DBI
        - path_loss_db
    )

    # Receiver noise power (dBm)
    noise_power_dbm = (
        THERMAL_NOISE_DBM_HZ
        + 10 * math.log10(BANDWIDTH_HZ)
        + NOISE_FIGURE_DB
    )

    # Signal-to-noise ratio (dB)
    snr_db = received_power_dbm - noise_power_dbm

    return {
        "ground_distance_km": ground_distance_km,
        "path_loss_db": path_loss_db,
        "received_power_dbm": received_power_dbm,
        "noise_power_dbm": noise_power_dbm,
        "snr_db": snr_db
    }


if __name__ == "__main__":
    for distance in [10, 20, 50, 100]:
        result = calculate_link_budget(distance)

        print(
            f"Distance: {distance:6.1f} km | "
            f"Rx Power: {result['received_power_dbm']:8.3f} dBm | "
            f"SNR: {result['snr_db']:8.3f} dB"
        )