
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut


def train_model():

    df = pd.read_csv("data/haps_dataset.csv")

    features = [
        "ground_distance_km",
        "slant_range_km",
        "elevation_deg"
    ]

    logo = LeaveOneGroupOut()

    all_results = []

    print("Starting HAPS prediction model...")

    for scenario in df["scenario"].unique():

        scenario_df = df[df["scenario"] == scenario].copy()

        X = scenario_df[features]
        y = scenario_df["snr_db"]
        groups = scenario_df["trajectory_id"]

        actual_values = []
        predicted_values = []

        for train_idx, test_idx in logo.split(X, y, groups):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            actual_values.extend(y_test.tolist())
            predicted_values.extend(predictions.tolist())

            all_results.extend([
    {
        "scenario": scenario,
        "trajectory_id": groups.iloc[idx],
        "position_index": scenario_df.iloc[idx]["position_index"],
        "actual_snr_db": actual,
        "predicted_snr_db": predicted
    }
                for idx, actual, predicted in zip(
                    test_idx,
                    y_test.tolist(),
                    predictions.tolist()
                )
            ])

        mae = mean_absolute_error(
            actual_values,
            predicted_values
        )

        print("\nScenario:", scenario)
        print("Samples:", len(scenario_df))
        print("Trajectories:", groups.nunique())
        print("Prediction MAE:", round(mae, 4), "dB")

    results = pd.DataFrame(all_results)

    results.to_csv(
        "data/prediction_results.csv",
        index=False
    )

    print("\nPrediction results saved successfully!")


if __name__ == "__main__":
    train_model()

