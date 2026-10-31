import math

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

from src.dataset_generator import generate_dataset
from src.environment import calculate_rain_attenuation
from src.geometry import calculate_geometry
from src.ml_model import FEATURES
from src.propagation import calculate_path_loss
from src.sensitivity_analysis import (
    REFERENCE_GROUND_DISTANCE_KM,
    build_sensitivity_results,
    reference_propagation,
)


def test_clear_and_rain_reference_scenarios():
    clear = reference_propagation(50.0, "clear_air")
    rain = reference_propagation(50.0, "rain_affected")

    assert math.isclose(clear["basic_path_loss_db"], calculate_path_loss(50.0))
    assert clear["rain_attenuation_db"] == 0.0
    assert rain["rain_attenuation_db"] > 0.0
    assert math.isclose(
        rain["path_loss_db"] - clear["path_loss_db"],
        rain["rain_attenuation_db"],
        abs_tol=1e-12,
    )


def test_dataset_applies_rain_attenuation_to_loss_power_and_snr():
    dataset = generate_dataset()
    clear = dataset[dataset["scenario"] == "clear_air"].set_index(
        ["trajectory_id", "position_index"]
    )
    rain = dataset[dataset["scenario"] == "rain_affected"].set_index(
        ["trajectory_id", "position_index"]
    )

    assert (rain["rain_attenuation_db"] > 0).all()
    pd.testing.assert_series_equal(
        rain["path_loss_db"] - clear["path_loss_db"],
        rain["rain_attenuation_db"],
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        clear["snr_db"] - rain["snr_db"],
        rain["rain_attenuation_db"],
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_analytical_baseline_mae_is_reproducible():
    dataset = pd.read_csv("data/haps_dataset.csv")
    predictions = pd.read_csv("data/prediction_results.csv")
    merged = dataset.merge(
        predictions,
        on=["scenario", "trajectory_id", "position_index"],
    )

    expected = {
        "clear_air": 0.013775304768091171,
        "rain_affected": 0.013698346123443719,
    }
    for scenario, group in merged.groupby("scenario"):
        mae = mean_absolute_error(group["snr_db"], group["predicted_snr_db"])
        assert math.isclose(mae, expected[scenario], rel_tol=0, abs_tol=1e-12)


def test_leave_one_group_out_holds_out_complete_trajectories():
    dataset = pd.read_csv("data/haps_dataset.csv")
    scenario = dataset[dataset["scenario"] == "clear_air"]
    logo = LeaveOneGroupOut()
    held_out = set()

    for train_idx, test_idx in logo.split(
        scenario[FEATURES],
        scenario["snr_db"],
        scenario["trajectory_id"],
    ):
        train_groups = set(scenario.iloc[train_idx]["trajectory_id"])
        test_groups = set(scenario.iloc[test_idx]["trajectory_id"])
        assert train_groups.isdisjoint(test_groups)
        assert len(test_groups) == 1
        held_out.update(test_groups)

    assert held_out == set(dataset["trajectory_id"].unique())


def test_ml_features_do_not_include_target_or_derived_outputs():
    forbidden = {"snr_db", "path_loss_db", "rain_attenuation_db"}
    assert forbidden.isdisjoint(FEATURES)


def test_sensitivity_reference_provenance_and_separation():
    reference = reference_propagation()
    results = build_sensitivity_results()

    assert reference["ground_distance_km"] == REFERENCE_GROUND_DISTANCE_KM
    assert reference["scenario"] == "clear_air"
    assert math.isclose(
        reference["basic_path_loss_db"],
        calculate_path_loss(REFERENCE_GROUND_DISTANCE_KM),
        abs_tol=1e-12,
    )
    assert {"link_budget", "propagation"} == set(results["parameter_type"])

    link_rows = results[results["parameter_type"] == "link_budget"]
    propagation_rows = results[results["parameter_type"] == "propagation"]
    assert link_rows["path_loss_db"].nunique() == 1
    assert propagation_rows["path_loss_db"].nunique() > 1


def test_reference_rain_uses_existing_rain_model():
    _, elevation = calculate_geometry(REFERENCE_GROUND_DISTANCE_KM)
    expected = calculate_rain_attenuation(2.0, elevation)
    observed = reference_propagation(
        REFERENCE_GROUND_DISTANCE_KM,
        "rain_affected",
    )["rain_attenuation_db"]
    assert math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12)
