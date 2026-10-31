
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

    metric_rows = []

    for scenario in comparison["scenario"].unique():

        subset = comparison[
            comparison["scenario"] == scenario
        ]

        analytical_received_power = subset["received_power_dbm"]
        received_power_predictions = subset["predicted_received_power_dbm"]
        analytical_snr = subset["snr_db"]
        snr_predictions = subset["predicted_snr_db"]

        received_power_mae = mean_absolute_error(
            analytical_received_power,
            received_power_predictions,
        )
        snr_mae = mean_absolute_error(
            analytical_snr,
            snr_predictions,
        )

        print("\nScenario:", scenario)
        print("Samples:", len(subset))
        print("Analytical reference: P.528 + link budget")
        print(
            "Received-power ML MAE:",
            round(received_power_mae, 4),
            "dB",
        )
        print("SNR ML MAE:", round(snr_mae, 4), "dB")
        metric_rows.append({
            "scenario": scenario,
            "samples": len(subset),
            "received_power_mae_db": received_power_mae,
            "snr_mae_db": snr_mae,
        })

    comparison.to_csv(
        "data/baseline_comparison.csv",
        index=False
    )

    print("\nComparison saved to data/baseline_comparison.csv")
    return comparison, pd.DataFrame(metric_rows)


if __name__ == "__main__":
    compare_baseline()

