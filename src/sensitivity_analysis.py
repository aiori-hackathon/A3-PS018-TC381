"""Reproducible sensitivity analysis for the PS-018 link budget.

The reference path loss is always obtained from the configured P.528 model.
Link-budget sensitivities hold that propagation result fixed, while
propagation sensitivities recompute it for each ground distance. Rain loss is
calculated by the existing P.838 helper and is kept separate from basic
transmission loss.
"""

import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.environment import (
    calculate_rain_attenuation,
    RAINFALL_RATE_MM_H,
    RAIN_LAYER_HEIGHT_KM,
)
from src.geometry import calculate_geometry
from src.link_budget import (
    BANDWIDTH_HZ,
    NOISE_FIGURE_DB,
    RX_GAIN_DBI,
    THERMAL_NOISE_DBM_HZ,
    TX_GAIN_DBI,
    TX_POWER_DBM,
)
from src.propagation import (
    FREQUENCY_MHZ,
    HAPS_HEIGHT_M,
    POLARIZATION,
    RECEIVER_HEIGHT_M,
    TIME_PERCENTAGE,
    calculate_path_loss,
)


OUTPUT_PATH = Path("data/sensitivity_results.csv")
PLOT_PATH = Path("data/sensitivity_analysis.png")

REFERENCE_GROUND_DISTANCE_KM = 50.0
REFERENCE_SCENARIO = "clear_air"
REFERENCE_FREQUENCY_GHZ = FREQUENCY_MHZ / 1000.0


def calculate_snr(
    tx_power,
    tx_gain,
    rx_gain,
    bandwidth,
    noise_figure,
    path_loss,
):
    """Calculate SNR from link-budget parameters and total path loss."""

    received_power = tx_power + tx_gain + rx_gain - path_loss
    noise_power = (
        THERMAL_NOISE_DBM_HZ
        + 10 * math.log10(bandwidth)
        + noise_figure
    )
    return received_power - noise_power


def reference_propagation(
    ground_distance_km=REFERENCE_GROUND_DISTANCE_KM,
    scenario=REFERENCE_SCENARIO,
):
    """Return provenance and total loss for one reproducible reference case.

    ``basic_path_loss_db`` comes directly from P.528. For the rain scenario,
    the existing P.838-based attenuation is added explicitly; no alternate
    propagation equation is introduced here.
    """

    if scenario not in {"clear_air", "rain_affected"}:
        raise ValueError(f"Unsupported scenario: {scenario}")

    basic_path_loss_db = calculate_path_loss(ground_distance_km)
    _, elevation_deg = calculate_geometry(ground_distance_km)
    rain_attenuation_db = 0.0
    if scenario == "rain_affected":
        rain_attenuation_db = calculate_rain_attenuation(
            frequency_ghz=REFERENCE_FREQUENCY_GHZ,
            elevation_deg=elevation_deg,
        )

    return {
        "ground_distance_km": float(ground_distance_km),
        "scenario": scenario,
        "frequency_ghz": REFERENCE_FREQUENCY_GHZ,
        "receiver_height_m": RECEIVER_HEIGHT_M,
        "haps_height_m": HAPS_HEIGHT_M,
        "time_percentage": TIME_PERCENTAGE,
        "polarization": POLARIZATION,
        "rainfall_rate_mm_h": RAINFALL_RATE_MM_H,
        "rain_layer_height_km": RAIN_LAYER_HEIGHT_KM,
        "basic_path_loss_db": float(basic_path_loss_db),
        "rain_attenuation_db": float(rain_attenuation_db),
        "path_loss_db": float(basic_path_loss_db + rain_attenuation_db),
        "elevation_deg": float(elevation_deg),
    }


def _result_row(
    parameter,
    parameter_type,
    value,
    value_unit,
    link_budget,
    propagation,
):
    """Create one fully-provenanced result row."""

    return {
        "parameter": parameter,
        "parameter_type": parameter_type,
        "value": value,
        "value_unit": value_unit,
        "tx_power_dbm": link_budget["tx_power_dbm"],
        "tx_gain_dbi": link_budget["tx_gain_dbi"],
        "rx_gain_dbi": link_budget["rx_gain_dbi"],
        "bandwidth_hz": link_budget["bandwidth_hz"],
        "noise_figure_db": link_budget["noise_figure_db"],
        **propagation,
        "snr_db": calculate_snr(
            link_budget["tx_power_dbm"],
            link_budget["tx_gain_dbi"],
            link_budget["rx_gain_dbi"],
            link_budget["bandwidth_hz"],
            link_budget["noise_figure_db"],
            propagation["path_loss_db"],
        ),
    }


def build_sensitivity_results():
    """Build deterministic link-budget and propagation sensitivity rows."""

    reference = reference_propagation()
    base_link_budget = {
        "tx_power_dbm": TX_POWER_DBM,
        "tx_gain_dbi": TX_GAIN_DBI,
        "rx_gain_dbi": RX_GAIN_DBI,
        "bandwidth_hz": BANDWIDTH_HZ,
        "noise_figure_db": NOISE_FIGURE_DB,
    }

    rows = []
    link_budget_cases = {
        "Transmit Power": ([25.0, 30.0, 35.0], "dBm", "tx_power_dbm"),
        "Transmit Gain": ([15.0, 20.0, 25.0], "dBi", "tx_gain_dbi"),
        "Receive Gain": ([0.0, 5.0, 10.0], "dBi", "rx_gain_dbi"),
        "Bandwidth": ([5e6, 10e6, 20e6], "Hz", "bandwidth_hz"),
        "Noise Figure": ([3.0, 5.0, 7.0], "dB", "noise_figure_db"),
    }
    for parameter, (values, unit, key) in link_budget_cases.items():
        for value in values:
            link_budget = dict(base_link_budget)
            link_budget[key] = value
            rows.append(
                _result_row(
                    parameter,
                    "link_budget",
                    value,
                    unit,
                    link_budget,
                    reference,
                )
            )

    for distance in [25.0, REFERENCE_GROUND_DISTANCE_KM, 75.0]:
        propagation = reference_propagation(distance, "clear_air")
        rows.append(
            _result_row(
                "Reference Ground Distance",
                "propagation",
                distance,
                "km",
                base_link_budget,
                propagation,
            )
        )

    for scenario in ["clear_air", "rain_affected"]:
        propagation = reference_propagation(
            REFERENCE_GROUND_DISTANCE_KM,
            scenario,
        )
        rows.append(
            _result_row(
                "Environmental Scenario",
                "propagation",
                scenario,
                "scenario",
                base_link_budget,
                propagation,
            )
        )

    return pd.DataFrame(rows)


def save_sensitivity_plot(df, output_path=PLOT_PATH):
    """Save a compact plot of each sensitivity family."""

    parameters = df["parameter"].unique()
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()
    for axis, parameter in zip(axes, parameters):
        subset = df[df["parameter"] == parameter]
        if parameter == "Environmental Scenario":
            axis.bar(subset["value"], subset["snr_db"])
            axis.tick_params(axis="x", rotation=25)
        else:
            axis.plot(
                subset["value"].astype(float),
                subset["snr_db"],
                marker="o",
            )
        axis.set_title(parameter)
        axis.set_ylabel("SNR (dB)")
        axis.grid(True, alpha=0.3)
    for axis in axes[len(parameters):]:
        axis.set_visible(False)
    figure.suptitle("PS-018 sensitivity analysis")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_sensitivity_analysis():
    """Write reproducible sensitivity data and a plot from the repo root."""

    results = build_sensitivity_results()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    save_sensitivity_plot(results)

    print("\nSENSITIVITY ANALYSIS")
    print("=" * 40)
    print(f"Reference: {REFERENCE_GROUND_DISTANCE_KM:g} km, {REFERENCE_SCENARIO}")
    print(
        f"P.528 basic path loss: "
        f"{reference_propagation()['basic_path_loss_db']:.5f} dB"
    )
    print(f"Saved to {OUTPUT_PATH} and {PLOT_PATH}")
    return results


if __name__ == "__main__":
    run_sensitivity_analysis()
