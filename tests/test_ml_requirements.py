import math

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from src.inference_benchmark import benchmark_inference
from src.ml_model import FEATURES, evaluate_models


DATASET_PATH = "data/haps_dataset.csv"
PREDICTIONS_PATH = "data/prediction_results.csv"
BENCHMARK_PATH = "data/inference_benchmark.csv"


def test_prediction_artifact_has_received_power_primary_and_snr_secondary():
    dataset = pd.read_csv(DATASET_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    required = {
        "scenario",
        "trajectory_id",
        "position_index",
        "actual_received_power_dbm",
        "predicted_received_power_dbm",
        "actual_snr_db",
        "predicted_snr_db",
    }
    assert required.issubset(predictions.columns)
    assert predictions["predicted_received_power_dbm"].notna().all()
    assert predictions["predicted_snr_db"].notna().all()

    merged = dataset.merge(
        predictions,
        on=["scenario", "trajectory_id", "position_index"],
    )
    assert np.allclose(
        merged["actual_received_power_dbm"],
        merged["received_power_dbm"],
        rtol=0,
        atol=1e-12,
    )
    assert np.allclose(
        merged["actual_snr_db"],
        merged["snr_db"],
        rtol=0,
        atol=1e-12,
    )
    assert "received_power_dbm" not in FEATURES
    assert "snr_db" not in FEATURES


def test_received_power_and_snr_mae_match_saved_baseline():
    comparison = pd.read_csv("data/baseline_comparison.csv")
    expected = {
        "clear_air": (0.013470933920770612, 0.013775304768091171),
        "rain_affected": (0.013035060132768103, 0.013698346123443719),
    }
    for scenario, subset in comparison.groupby("scenario"):
        received_mae = mean_absolute_error(
            subset["received_power_dbm"],
            subset["predicted_received_power_dbm"],
        )
        snr_mae = mean_absolute_error(
            subset["snr_db"],
            subset["predicted_snr_db"],
        )
        assert math.isclose(received_mae, expected[scenario][0], abs_tol=1e-12)
        assert math.isclose(snr_mae, expected[scenario][1], abs_tol=1e-12)


def test_evaluation_returns_complete_trajectory_held_out_predictions():
    dataset = pd.read_csv(DATASET_PATH)
    predictions, metrics = evaluate_models(dataset)
    assert len(predictions) == len(dataset)
    assert len(metrics) == 2
    for scenario in dataset["scenario"].unique():
        expected_groups = set(dataset[dataset["scenario"] == scenario]["trajectory_id"])
        actual_groups = set(predictions[predictions["scenario"] == scenario]["trajectory_id"])
        assert actual_groups == expected_groups


def test_inference_benchmark_artifact_structure():
    benchmark = pd.read_csv(BENCHMARK_PATH)
    required = {
        "target",
        "sample_count",
        "repetitions",
        "total_prediction_time_s",
        "per_sample_prediction_time_ms",
        "python_version",
        "scikit_learn_version",
        "model_loading_excluded",
        "model_training_excluded",
        "timed_operation",
    }
    assert required.issubset(benchmark.columns)
    assert set(benchmark["target"]) == {"received_power_dbm", "snr_db"}
    assert (benchmark["sample_count"] == 470).all()
    assert (benchmark["repetitions"] >= 1).all()
    assert (benchmark["total_prediction_time_s"] > 0).all()
    assert (benchmark["per_sample_prediction_time_ms"] > 0).all()
    assert benchmark["model_loading_excluded"].all()
    assert benchmark["model_training_excluded"].all()
    assert (benchmark["timed_operation"] == "RandomForestRegressor.predict").all()


def test_inference_benchmark_rejects_invalid_repetitions(tmp_path):
    dataset = pd.read_csv(DATASET_PATH)
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    try:
        benchmark_inference(dataset_path, tmp_path / "benchmark.csv", repetitions=0)
    except ValueError as error:
        assert "Repetitions" in str(error)
    else:
        raise AssertionError("Expected invalid repetitions to be rejected")
