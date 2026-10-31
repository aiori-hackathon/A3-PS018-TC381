"""Prediction-only inference benchmark for the PS-018 ML models."""

import platform
import sys
import time
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import LeaveOneGroupOut

from src.ml_model import FEATURES, TARGETS, _new_model


OUTPUT_PATH = Path("data/inference_benchmark.csv")
DEFAULT_REPETITIONS = 20


def _fit_held_out_models(dataset):
    """Fit grouped models before timing and return held-out prediction jobs."""

    jobs = []
    logo = LeaveOneGroupOut()
    for scenario in dataset["scenario"].unique():
        scenario_df = dataset[dataset["scenario"] == scenario].copy()
        X = scenario_df[FEATURES]
        groups = scenario_df["trajectory_id"]
        for train_idx, test_idx in logo.split(X, groups=groups):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            for target in TARGETS:
                model = _new_model()
                model.fit(X_train, scenario_df.iloc[train_idx][target])
                jobs.append({
                    "scenario": scenario,
                    "target": target,
                    "model": model,
                    "X_test": X_test,
                    "samples": len(test_idx),
                })
    return jobs


def benchmark_inference(
    dataset_path=Path("data/haps_dataset.csv"),
    output_path=OUTPUT_PATH,
    repetitions=DEFAULT_REPETITIONS,
):
    """Measure repeated held-out ``predict`` calls, excluding fit and loading."""

    if repetitions < 1:
        raise ValueError("Repetitions must be at least one.")

    dataset = pd.read_csv(dataset_path)
    jobs = _fit_held_out_models(dataset)
    rows = []
    for target in TARGETS:
        target_jobs = [job for job in jobs if job["target"] == target]
        sample_count = sum(job["samples"] for job in target_jobs)
        start = time.perf_counter()
        for _ in range(repetitions):
            for job in target_jobs:
                # Only this prediction call is inside the timed region.
                job["model"].predict(job["X_test"])
        elapsed_s = time.perf_counter() - start
        total_predictions = sample_count * repetitions
        rows.append({
            "target": target,
            "target_label": TARGETS[target],
            "scenarios": dataset["scenario"].nunique(),
            "held_out_folds": len(target_jobs),
            "sample_count": sample_count,
            "repetitions": repetitions,
            "total_prediction_time_s": elapsed_s,
            "per_sample_prediction_time_ms": (
                elapsed_s / total_predictions * 1000.0
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "model_loading_excluded": True,
            "model_training_excluded": True,
            "timed_operation": "RandomForestRegressor.predict",
        })

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


if __name__ == "__main__":
    result = benchmark_inference()
    print(result.to_string(index=False))
    print(f"\nBenchmark saved to {OUTPUT_PATH}")
