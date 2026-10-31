"""Leave-one-trajectory-out ML evaluation for PS-018.

Received power is the primary PS-018 prediction target. SNR is evaluated by a
second model using the same grouped splits and remains a useful secondary
link-quality result. The trajectories share broad ground-distance ranges, so
held-out samples are mostly interpolation within the same synthetic geometry
domain; the MAE values do not establish real-world generalization.
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut


FEATURES = [
    "ground_distance_km",
    "slant_range_km",
    "elevation_deg",
]

TARGETS = {
    "received_power_dbm": "Received power (dBm)",
    "snr_db": "SNR (dB)",
}

MODEL_KWARGS = {
    "n_estimators": 100,
    "random_state": 42,
}


def _new_model():
    return RandomForestRegressor(**MODEL_KWARGS)


def evaluate_models(df):
    """Return held-out predictions and per-scenario MAE for both targets."""

    logo = LeaveOneGroupOut()
    all_results = []
    metric_rows = []

    for scenario in df["scenario"].unique():
        scenario_df = df[df["scenario"] == scenario].copy()
        X = scenario_df[FEATURES]
        groups = scenario_df["trajectory_id"]
        actual_values = {target: [] for target in TARGETS}
        predicted_values = {target: [] for target in TARGETS}

        for train_idx, test_idx in logo.split(X, groups=groups):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            models = {}

            for target in TARGETS:
                model = _new_model()
                model.fit(X_train, scenario_df.iloc[train_idx][target])
                models[target] = model

            predictions = {
                target: models[target].predict(X_test)
                for target in TARGETS
            }

            for target in TARGETS:
                actual = scenario_df.iloc[test_idx][target].tolist()
                predicted = predictions[target].tolist()
                actual_values[target].extend(actual)
                predicted_values[target].extend(predicted)

            for local_position, row_index in enumerate(test_idx):
                row = scenario_df.iloc[row_index]
                all_results.append({
                    "scenario": scenario,
                    "trajectory_id": row["trajectory_id"],
                    "position_index": row["position_index"],
                    "actual_received_power_dbm": row["received_power_dbm"],
                    "predicted_received_power_dbm": predictions[
                        "received_power_dbm"
                    ][local_position],
                    "actual_snr_db": row["snr_db"],
                    "predicted_snr_db": predictions["snr_db"][local_position],
                })

        received_mae = mean_absolute_error(
            actual_values["received_power_dbm"],
            predicted_values["received_power_dbm"],
        )
        snr_mae = mean_absolute_error(
            actual_values["snr_db"],
            predicted_values["snr_db"],
        )
        metric_rows.append({
            "scenario": scenario,
            "samples": len(scenario_df),
            "trajectories": groups.nunique(),
            "received_power_mae_db": received_mae,
            "snr_mae_db": snr_mae,
        })

    return pd.DataFrame(all_results), pd.DataFrame(metric_rows)


def train_model(
    dataset_path=Path("data/haps_dataset.csv"),
    output_path=Path("data/prediction_results.csv"),
):
    """Train grouped models, save both prediction targets, and report MAE."""

    df = pd.read_csv(dataset_path)
    results, metrics = evaluate_models(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("Starting HAPS prediction model...")
    for row in metrics.itertuples(index=False):
        print(f"\nScenario: {row.scenario}")
        print("Samples:", row.samples)
        print("Trajectories:", row.trajectories)
        print("Received-power MAE:", round(row.received_power_mae_db, 4), "dB")
        print("SNR MAE:", round(row.snr_mae_db, 4), "dB")
    print(f"\nPrediction results saved to {output_path}")
    return results, metrics


if __name__ == "__main__":
    train_model()
