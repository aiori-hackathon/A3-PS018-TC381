"""Streamlit demonstration for the AIORI PS-018 HAPS link model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.environment import calculate_rain_attenuation
from src.geometry import calculate_geometry
from src.link_budget import (
    BANDWIDTH_HZ,
    NOISE_FIGURE_DB,
    RX_GAIN_DBI,
    TX_GAIN_DBI,
    TX_POWER_DBM,
    calculate_link_budget_for_parameters,
)
from src.propagation import (
    FREQUENCY_MHZ,
    HAPS_HEIGHT_M,
    RECEIVER_HEIGHT_M,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "haps_dataset.csv"
PREDICTIONS_PATH = DATA_DIR / "prediction_results.csv"
BENCHMARK_PATH = DATA_DIR / "inference_benchmark.csv"
SENSITIVITY_PATH = DATA_DIR / "sensitivity_results.csv"
SENSITIVITY_PLOT_PATH = DATA_DIR / "sensitivity_analysis.png"
FREQUENCY_GHZ = FREQUENCY_MHZ / 1000.0


def calculate_demo_outputs(
    ground_distance_km,
    tx_power_dbm=TX_POWER_DBM,
    tx_gain_dbi=TX_GAIN_DBI,
    rx_gain_dbi=RX_GAIN_DBI,
    noise_figure_db=NOISE_FIGURE_DB,
    bandwidth_hz=BANDWIDTH_HZ,
    scenario="clear_air",
):
    """Calculate the dashboard values using the project model functions."""

    if scenario not in {"clear_air", "rain_affected"}:
        raise ValueError("Scenario must be clear_air or rain_affected.")

    slant_range_km, elevation_deg = calculate_geometry(ground_distance_km)
    rain_attenuation_db = 0.0
    if scenario == "rain_affected":
        rain_attenuation_db = calculate_rain_attenuation(
            frequency_ghz=FREQUENCY_GHZ,
            elevation_deg=elevation_deg,
        )

    link = calculate_link_budget_for_parameters(
        ground_distance_km=ground_distance_km,
        tx_power_dbm=tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        bandwidth_hz=bandwidth_hz,
        noise_figure_db=noise_figure_db,
        additional_loss_db=rain_attenuation_db,
    )
    return {
        "slant_range_km": slant_range_km,
        "elevation_deg": elevation_deg,
        "rain_attenuation_db": rain_attenuation_db,
        **link,
    }


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_benchmark():
    return pd.read_csv(BENCHMARK_PATH)


@st.cache_data
def load_sensitivity_results():
    return pd.read_csv(SENSITIVITY_PATH)


def make_snr_distance_plot(dataset, selected_scenario, selected_distance, selected_snr):
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for scenario, colour in [("clear_air", "#1f77b4"), ("rain_affected", "#d95f02")]:
        series = (
            dataset[dataset["scenario"] == scenario]
            .sort_values("ground_distance_km")
            .drop_duplicates("ground_distance_km")
        )
        axis.plot(
            series["ground_distance_km"],
            series["snr_db"],
            label=scenario.replace("_", " ").title(),
            color=colour,
        )
    axis.scatter(
        [selected_distance],
        [selected_snr],
        color="black",
        marker="*",
        s=120,
        label="Current controls",
        zorder=3,
    )
    axis.set_xlabel("Ground distance (km)")
    axis.set_ylabel("Analytical SNR (dB)")
    axis.set_title(
        "Baseline analytical SNR by scenario; marker uses current controls"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def make_prediction_plot(merged_predictions, target="received_power"):
    if target == "received_power":
        actual_column = "received_power_dbm"
        predicted_column = "predicted_received_power_dbm"
        x_label = "Analytical received power (dBm)"
        y_label = "Precomputed ML received power (dBm)"
        title = "Analytical versus precomputed received-power predictions"
    elif target == "snr":
        actual_column = "snr_db"
        predicted_column = "predicted_snr_db"
        x_label = "Analytical SNR (dB)"
        y_label = "Precomputed ML SNR (dB)"
        title = "Analytical versus precomputed SNR predictions"
    else:
        raise ValueError("Unsupported prediction target.")

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for scenario, colour in [("clear_air", "#1f77b4"), ("rain_affected", "#d95f02")]:
        subset = merged_predictions[merged_predictions["scenario"] == scenario]
        axis.scatter(
            subset[actual_column],
            subset[predicted_column],
            s=16,
            alpha=0.65,
            color=colour,
            label=scenario.replace("_", " ").title(),
        )
    limits = [
        merged_predictions[[actual_column, predicted_column]].min().min(),
        merged_predictions[[actual_column, predicted_column]].max().max(),
    ]
    axis.plot(limits, limits, "k--", label="Perfect agreement")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def make_sensitivity_plot(sensitivity):
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    axes = axes.ravel()
    parameters = sensitivity["parameter"].unique()
    for axis, parameter in zip(axes, parameters):
        subset = sensitivity[sensitivity["parameter"] == parameter]
        if parameter == "Environmental Scenario":
            axis.bar(subset["value"], subset["snr_db"], color="#6a3d9a")
            axis.tick_params(axis="x", rotation=25)
        else:
            axis.plot(
                subset["value"].astype(float),
                subset["snr_db"],
                marker="o",
                color="#6a3d9a",
            )
        axis.set_title(parameter, fontsize=9)
        axis.set_ylabel("SNR (dB)")
        axis.grid(True, alpha=0.25)
    for axis in axes[len(parameters):]:
        axis.set_visible(False)
    return figure


def main():
    st.set_page_config(
        page_title="PS-018 HAPS Link Quality",
        page_icon="📡",
        layout="wide",
    )
    st.title("📡 PS-018 HAPS Link-Quality Demonstration")
    st.caption(
        "An interactive view of the documented HAPS propagation, link-budget, "
        "rain-scenario, and precomputed ML benchmark."
    )

    with st.sidebar:
        st.header("Link controls")
        ground_distance_km = st.slider(
            "Ground distance (km)", 1.0, 100.0, 50.0, 1.0,
            help="Surface distance from the receiver to the point below the HAPS.",
        )
        scenario = st.radio(
            "Environmental scenario",
            ["clear_air", "rain_affected"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        st.subheader("Communication parameters")
        tx_power_dbm = st.number_input("Transmit power (dBm)", 0.0, 60.0, float(TX_POWER_DBM), 1.0)
        tx_gain_dbi = st.number_input("Transmit antenna gain (dBi)", 0.0, 50.0, float(TX_GAIN_DBI), 1.0)
        rx_gain_dbi = st.number_input("Receive antenna gain (dBi)", 0.0, 50.0, float(RX_GAIN_DBI), 1.0)
        noise_figure_db = st.number_input("Receiver noise figure (dB)", 0.0, 30.0, float(NOISE_FIGURE_DB), 0.5)
        bandwidth_mhz = st.number_input("Bandwidth (MHz)", 0.1, 1000.0, BANDWIDTH_HZ / 1e6, 0.5)

    try:
        outputs = calculate_demo_outputs(
            ground_distance_km,
            tx_power_dbm,
            tx_gain_dbi,
            rx_gain_dbi,
            noise_figure_db,
            bandwidth_mhz * 1e6,
            scenario,
        )
        dataset = load_dataset()
        predictions = load_predictions()
        benchmark = load_benchmark()
        sensitivity = load_sensitivity_results()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        st.error(f"The selected calculation could not be completed: {error}")
        st.stop()

    st.info(
        "The dashboard uses the existing ITU-R P.528-5 propagation model. "
        "Rain attenuation is the project’s simplified P.838-based effective "
        "rain-path approximation, intended for scenario comparison—not a full "
        "atmospheric rain-path prediction."
    )

    st.header("Current link result")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Received power", f"{outputs['received_power_dbm']:.2f} dBm")
    metric_columns[1].metric("SNR", f"{outputs['snr_db']:.2f} dB")
    metric_columns[2].metric("Total path loss", f"{outputs['path_loss_db']:.2f} dB")
    metric_columns[3].metric("Environmental attenuation", f"{outputs['rain_attenuation_db']:.3f} dB")

    geometry_columns = st.columns(4)
    geometry_columns[0].metric("HAPS altitude", f"{HAPS_HEIGHT_M / 1000:.1f} km")
    geometry_columns[1].metric("Receiver height", f"{RECEIVER_HEIGHT_M:.1f} m")
    geometry_columns[2].metric("Ground distance", f"{ground_distance_km:.1f} km")
    geometry_columns[3].metric("Calculated slant distance", f"{outputs['slant_range_km']:.2f} km")
    st.caption(f"Calculated elevation angle: {outputs['elevation_deg']:.2f}°")

    st.header("Model views")
    st.pyplot(
        make_snr_distance_plot(
            dataset, scenario, ground_distance_km, outputs["snr_db"]
        ),
        clear_figure=True,
    )

    merged_predictions = dataset.merge(
        predictions,
        on=["scenario", "trajectory_id", "position_index"],
    )
    nearest = merged_predictions[merged_predictions["scenario"] == scenario].copy()
    nearest["distance_error"] = (nearest["ground_distance_km"] - ground_distance_km).abs()
    nearest_row = nearest.loc[nearest["distance_error"].idxmin()]
    st.subheader("Primary ML comparison: received power")
    st.warning(
        "ML values below are precomputed baseline received-power results in "
        "data/prediction_results.csv using Leave-One-Trajectory-Out evaluation. "
        "They use the checked-in dataset’s default link parameters, so changing "
        "the controls does not change these baseline values. The dashboard does "
        "not retrain a model live, and these synthetic results are not real-world "
        "validation."
    )
    ml_columns = st.columns(3)
    ml_columns[0].metric("Baseline analytical received power", f"{nearest_row['received_power_dbm']:.3f} dBm")
    ml_columns[1].metric("Baseline precomputed ML received power", f"{nearest_row['predicted_received_power_dbm']:.3f} dBm")
    ml_columns[2].metric("Nearest baseline distance", f"{nearest_row['ground_distance_km']:.1f} km")
    st.pyplot(make_prediction_plot(merged_predictions, "received_power"), clear_figure=True)
    st.caption(
        "Secondary link-quality result: the same held-out evaluation also predicts SNR."
    )
    secondary_columns = st.columns(2)
    secondary_columns[0].metric("Baseline analytical SNR", f"{nearest_row['snr_db']:.3f} dB")
    secondary_columns[1].metric("Baseline precomputed ML SNR", f"{nearest_row['predicted_snr_db']:.3f} dB")
    st.pyplot(make_prediction_plot(merged_predictions, "snr"), clear_figure=True)

    st.subheader("Prediction-only inference benchmark")
    st.caption(
        "Timing covers repeated RandomForestRegressor.predict calls on held-out "
        "samples. Model loading and training are excluded."
    )
    st.dataframe(
        benchmark[[
            "target_label", "sample_count", "repetitions",
            "total_prediction_time_s", "per_sample_prediction_time_ms",
            "python_version", "scikit_learn_version",
        ]],
        width="stretch",
        hide_index=True,
    )

    st.header("Verified sensitivity analysis")
    st.caption(
        "Link-budget variations hold the P.528 reference propagation result fixed. "
        "Propagation variations recompute path loss for each ground distance or scenario."
    )
    sensitivity_type = st.selectbox(
        "Show sensitivity family",
        ["All", "link_budget", "propagation"],
    )
    shown_sensitivity = sensitivity if sensitivity_type == "All" else sensitivity[
        sensitivity["parameter_type"] == sensitivity_type
    ]
    st.dataframe(
        shown_sensitivity[[
            "parameter", "parameter_type", "value", "value_unit",
            "ground_distance_km", "scenario", "basic_path_loss_db",
            "rain_attenuation_db", "path_loss_db", "snr_db",
        ]],
        width="stretch",
        hide_index=True,
    )
    st.pyplot(make_sensitivity_plot(shown_sensitivity), clear_figure=True)
    if SENSITIVITY_PLOT_PATH.exists():
        with st.expander("Open the saved Phase 2 sensitivity plot"):
            st.image(SENSITIVITY_PLOT_PATH)

    with st.expander("About this demonstration and its limitations"):
        st.markdown(
            """
            - HAPS altitude, receiver height, frequency, rainfall rate, and rain-layer height are the documented project assumptions.
            - The rain calculation uses the existing simplified effective-path approximation; it is not a complete atmospheric model.
            - The ML benchmark uses synthetic analytical labels and trajectories with overlapping distance ranges, so held-out predictions are mainly interpolation within this simulation domain.
            - No external service, live model training, or real-world measurement dataset is used.
            """
        )


if __name__ == "__main__":
    main()
