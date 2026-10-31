import math
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import (
    DATASET_PATH,
    PREDICTIONS_PATH,
    SENSITIVITY_PATH,
    calculate_demo_outputs,
    make_prediction_plot,
    make_sensitivity_plot,
    make_snr_distance_plot,
)
from src.environment import calculate_rain_attenuation
from src.geometry import calculate_geometry
from src.link_budget import calculate_link_budget_for_parameters


def metric_value(app_test, label):
    return next(metric.value for metric in app_test.metric if metric.label == label)


def test_dashboard_clear_air_and_rain_outputs_match_models():
    clear = calculate_demo_outputs(50.0, scenario="clear_air")
    rain = calculate_demo_outputs(50.0, scenario="rain_affected")
    slant, elevation = calculate_geometry(50.0)
    expected_rain = calculate_rain_attenuation(2.0, elevation)

    assert math.isclose(clear["slant_range_km"], slant, abs_tol=1e-12)
    assert math.isclose(clear["rain_attenuation_db"], 0.0, abs_tol=1e-12)
    assert math.isclose(rain["rain_attenuation_db"], expected_rain, abs_tol=1e-12)
    assert math.isclose(
        rain["path_loss_db"] - clear["path_loss_db"],
        expected_rain,
        abs_tol=1e-12,
    )
    assert math.isclose(
        clear["snr_db"] - rain["snr_db"],
        expected_rain,
        abs_tol=1e-12,
    )


def test_dashboard_controls_reach_link_budget_and_units_are_consistent():
    values = {
        "tx_power_dbm": 37.0,
        "tx_gain_dbi": 23.0,
        "rx_gain_dbi": 8.0,
        "noise_figure_db": 7.0,
        "bandwidth_hz": 25e6,
        "scenario": "rain_affected",
    }
    dashboard = calculate_demo_outputs(75.0, **values)
    expected = calculate_link_budget_for_parameters(
        ground_distance_km=75.0,
        additional_loss_db=dashboard["rain_attenuation_db"],
        **{key: value for key, value in values.items() if key != "scenario"},
    )
    assert dashboard["path_loss_db"] == expected["path_loss_db"]
    assert dashboard["received_power_dbm"] == expected["received_power_dbm"]
    assert dashboard["snr_db"] == expected["snr_db"]


def test_dashboard_boundary_and_invalid_inputs():
    for distance in [1.0, 100.0]:
        result = calculate_demo_outputs(distance)
        assert math.isfinite(result["snr_db"])
        assert result["slant_range_km"] > 0

    low_power = calculate_demo_outputs(
        1.0,
        tx_power_dbm=0.0,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        noise_figure_db=30.0,
        bandwidth_hz=100_000.0,
    )
    assert math.isfinite(low_power["snr_db"])

    with pytest.raises(ValueError):
        calculate_demo_outputs(50.0, bandwidth_hz=0.0)
    with pytest.raises(ValueError):
        calculate_demo_outputs(50.0, scenario="unknown")


def test_dashboard_streamlit_interactions_update_outputs():
    app_test = AppTest.from_file(
        Path(__file__).parents[1] / "app.py",
        default_timeout=30,
    ).run()
    assert not app_test.exception
    assert metric_value(app_test, "Environmental attenuation") == "0.000 dB"

    app_test.radio[0].set_value("rain_affected").run()
    assert not app_test.exception
    assert metric_value(app_test, "Environmental attenuation") != "0.000 dB"

    app_test.slider[0].set_value(100.0).run()
    assert metric_value(app_test, "Ground distance") == "100.0 km"
    assert metric_value(app_test, "SNR") == "15.16 dB"


def test_dashboard_visualizations_use_expected_data():
    dataset = pd.read_csv(DATASET_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    sensitivity = pd.read_csv(SENSITIVITY_PATH)
    merged = dataset.merge(
        predictions,
        on=["scenario", "trajectory_id", "position_index"],
    )

    snr_figure = make_snr_distance_plot(dataset, "clear_air", 50.0, 20.0)
    prediction_figure = make_prediction_plot(merged)
    snr_prediction_figure = make_prediction_plot(merged, "snr")
    sensitivity_figure = make_sensitivity_plot(sensitivity)
    try:
        assert len(snr_figure.axes[0].lines) == 2
        assert len(snr_figure.axes[0].collections) == 1
        assert len(prediction_figure.axes[0].collections) == 2
        assert len(prediction_figure.axes[0].lines) == 1
        assert prediction_figure.axes[0].get_xlabel() == "Analytical received power (dBm)"
        assert prediction_figure.axes[0].get_ylabel() == "Precomputed ML received power (dBm)"
        assert snr_prediction_figure.axes[0].get_xlabel() == "Analytical SNR (dB)"
        assert sum(axis.get_visible() for axis in sensitivity_figure.axes) == 7
    finally:
        import matplotlib.pyplot as plt

        plt.close(snr_figure)
        plt.close(prediction_figure)
        plt.close(snr_prediction_figure)
        plt.close(sensitivity_figure)
