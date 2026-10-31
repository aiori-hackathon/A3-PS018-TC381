
import pandas as pd

from src.link_budget import (
    TX_POWER_DBM,
    TX_GAIN_DBI,
    RX_GAIN_DBI,
    BANDWIDTH_HZ,
    NOISE_FIGURE_DB,
    THERMAL_NOISE_DBM_HZ
)

import math


def calculate_snr(tx_power, tx_gain, rx_gain,
                  bandwidth, noise_figure, path_loss):

    received_power = (
        tx_power + tx_gain + rx_gain - path_loss
    )

    noise_power = (
        THERMAL_NOISE_DBM_HZ
        + 10 * math.log10(bandwidth)
        + noise_figure
    )

    return received_power - noise_power


def run_sensitivity_analysis():

    path_loss = 133.14025

    parameters = {
        "Transmit Power (dBm)": [
            (25, TX_GAIN_DBI, RX_GAIN_DBI, BANDWIDTH_HZ, NOISE_FIGURE_DB),
            (30, TX_GAIN_DBI, RX_GAIN_DBI, BANDWIDTH_HZ, NOISE_FIGURE_DB),
            (35, TX_GAIN_DBI, RX_GAIN_DBI, BANDWIDTH_HZ, NOISE_FIGURE_DB)
        ],

        "Transmit Gain (dBi)": [
            (TX_POWER_DBM, 15, RX_GAIN_DBI, BANDWIDTH_HZ, NOISE_FIGURE_DB),
            (TX_POWER_DBM, 20, RX_GAIN_DBI, BANDWIDTH_HZ, NOISE_FIGURE_DB),
            (TX_POWER_DBM, 25, RX_GAIN_DBI, BANDWIDTH_HZ, NOISE_FIGURE_DB)
        ],

        "Receive Gain (dBi)": [
            (TX_POWER_DBM, TX_GAIN_DBI, 0, BANDWIDTH_HZ, NOISE_FIGURE_DB),
            (TX_POWER_DBM, TX_GAIN_DBI, 5, BANDWIDTH_HZ, NOISE_FIGURE_DB),
            (TX_POWER_DBM, TX_GAIN_DBI, 10, BANDWIDTH_HZ, NOISE_FIGURE_DB)
        ],

        "Bandwidth (MHz)": [
            (TX_POWER_DBM, TX_GAIN_DBI, RX_GAIN_DBI, 5e6, NOISE_FIGURE_DB),
            (TX_POWER_DBM, TX_GAIN_DBI, RX_GAIN_DBI, 10e6, NOISE_FIGURE_DB),
            (TX_POWER_DBM, TX_GAIN_DBI, RX_GAIN_DBI, 20e6, NOISE_FIGURE_DB)
        ],

        "Noise Figure (dB)": [
            (TX_POWER_DBM, TX_GAIN_DBI, RX_GAIN_DBI, BANDWIDTH_HZ, 3),
            (TX_POWER_DBM, TX_GAIN_DBI, RX_GAIN_DBI, BANDWIDTH_HZ, 5),
            (TX_POWER_DBM, TX_GAIN_DBI, RX_GAIN_DBI, BANDWIDTH_HZ, 7)
        ]
    }

    results = []

    for parameter, cases in parameters.items():

        for case in cases:

            snr = calculate_snr(
                *case,
                path_loss
            )

            results.append({
                "parameter": parameter,
                "snr_db": snr
            })

    df = pd.DataFrame(results)

    df.to_csv(
        "data/sensitivity_results.csv",
        index=False
    )

    print("\nSENSITIVITY ANALYSIS")
    print("=" * 40)

    for parameter, group in df.groupby("parameter"):
        print("\n", parameter)
        print(group["snr_db"].round(3).to_list())

    print("\nSaved to data/sensitivity_results.csv")


if __name__ == "__main__":
    run_sensitivity_analysis()

