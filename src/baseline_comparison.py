
import pandas as pd
from sklearn.metrics import mean_absolute_error


def compare_baseline():

    dataset = pd.read_csv("data/haps_dataset.csv")
    predictions = pd.read_csv("data/prediction_results.csv")

    comparison = dataset.merge(
        predictions,
        on=["scenario", "trajectory_id", "position_index"],
        suffixes=("_analytical", "_ml")
    )

    print("\nANALYTICAL BASELINE VS ML MODEL")
    print("=" * 45)

    for scenario in comparison["scenario"].unique():

        subset = comparison[
            comparison["scenario"] == scenario
        ]

        analytical_snr = subset["snr_db"]
        ml_predictions = subset["predicted_snr_db"]

        mae = mean_absolute_error(
            analytical_snr,
            ml_predictions
        )

        print("\nScenario:", scenario)
        print("Samples:", len(subset))
        print("Analytical reference: P.528 + link budget")
        print("ML prediction MAE:", round(mae, 4), "dB")

    comparison.to_csv(
        "data/baseline_comparison.csv",
        index=False
    )

    print("\nComparison saved to data/baseline_comparison.csv")


if __name__ == "__main__":
    compare_baseline()

