import math

from src.propagation import calculate_path_loss


# Link-budget assumptions
TX_POWER_DBM = 30.0
TX_GAIN_DBI = 20.0
RX_GAIN_DBI = 5.0

BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 5.0
THERMAL_NOISE_DBM_HZ = -174.0


def calculate_link_budget_for_parameters(
    ground_distance_km,
    tx_power_dbm=TX_POWER_DBM,
    tx_gain_dbi=TX_GAIN_DBI,
    rx_gain_dbi=RX_GAIN_DBI,
    bandwidth_hz=BANDWIDTH_HZ,
    noise_figure_db=NOISE_FIGURE_DB,
    additional_loss_db=0.0,
):
    """Calculate a link budget with explicit engineering parameters.

    ``additional_loss_db`` is reserved for environmental attenuation already
    calculated by an approved environmental model, such as the existing
    P.838-based rain helper. The basic P.528 path loss is always calculated
    here, so callers do not need to duplicate propagation equations.
    """

    if bandwidth_hz <= 0:
        raise ValueError("Bandwidth must be positive.")
    if not math.isfinite(additional_loss_db) or additional_loss_db < 0:
        raise ValueError("Additional loss must be finite and non-negative.")

    basic_path_loss_db = calculate_path_loss(ground_distance_km)
    path_loss_db = basic_path_loss_db + additional_loss_db

    # Received signal power (dBm)
    received_power_dbm = (
        tx_power_dbm
        + tx_gain_dbi
        + rx_gain_dbi
        - path_loss_db
    )

    # Receiver noise power (dBm)
    noise_power_dbm = (
        THERMAL_NOISE_DBM_HZ
        + 10 * math.log10(bandwidth_hz)
        + noise_figure_db
    )

    # Signal-to-noise ratio (dB)
    snr_db = received_power_dbm - noise_power_dbm

    return {
        "ground_distance_km": ground_distance_km,
        "basic_path_loss_db": basic_path_loss_db,
        "path_loss_db": path_loss_db,
        "additional_loss_db": additional_loss_db,
        "received_power_dbm": received_power_dbm,
        "noise_power_dbm": noise_power_dbm,
        "snr_db": snr_db
    }


def calculate_link_budget(ground_distance_km):
    """Calculate received power and SNR using project default parameters."""

    return calculate_link_budget_for_parameters(ground_distance_km)


if __name__ == "__main__":
    for distance in [10, 20, 50, 100]:
        result = calculate_link_budget(distance)

        print(
            f"Distance: {distance:6.1f} km | "
            f"Rx Power: {result['received_power_dbm']:8.3f} dBm | "
            f"SNR: {result['snr_db']:8.3f} dB"
        )
