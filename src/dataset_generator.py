from pathlib import Path

import pandas as pd

from src.trajectory import generate_trajectory
from src.link_budget import calculate_link_budget
from src.environment import calculate_rain_attenuation


OUTPUT_PATH = Path("data/haps_dataset.csv")
FREQUENCY_GHZ = 2.0


def generate_dataset():
    """
    Generate reproducible HAPS link data for two
    environmental scenarios.
    """

    rows = []

    trajectory_configs = [
        (10, 100, 5),
        (15, 100, 5),
        (10, 100, 4),
        (12, 100, 4),
        (10, 100, 2),
        (11, 100, 2),
        (10, 100, 3),
        (11.5, 100, 3),
    ]

    for trajectory_id, (start, end, step) in enumerate(
        trajectory_configs
    ):

        trajectory = generate_trajectory(
            start_km=start,
            end_km=end,
            step_km=step
        )

        for position_index, point in enumerate(trajectory):

            distance = point["ground_distance_km"]
            elevation = point["elevation_deg"]

            link = calculate_link_budget(distance)

            rain_loss = calculate_rain_attenuation(
                frequency_ghz=FREQUENCY_GHZ,
                elevation_deg=elevation
            )

            for scenario in ["clear_air", "rain_affected"]:

                additional_loss = (
                    0.0 if scenario == "clear_air"
                    else rain_loss
                )

                path_loss = (
                    link["path_loss_db"] + additional_loss
                )

                received_power = (
                    link["received_power_dbm"] - additional_loss
                )

                snr = (
                    link["snr_db"] - additional_loss
                )

                rows.append({
                    "trajectory_id": trajectory_id,
                    "position_index": position_index,
                    "scenario": scenario,
                    "ground_distance_km": distance,
                    "slant_range_km": point["slant_range_km"],
                    "elevation_deg": elevation,
                    "path_loss_db": path_loss,
                    "rain_attenuation_db": additional_loss,
                    "received_power_dbm": received_power,
                    "noise_power_dbm": link["noise_power_dbm"],
                    "snr_db": snr,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    dataset = generate_dataset()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(OUTPUT_PATH, index=False)

    print("Dataset generated successfully!")
    print("Total samples:", len(dataset))
    print(
        "Unique trajectories:",
        dataset["trajectory_id"].nunique()
    )
    print(
        "Environmental scenarios:",
        dataset["scenario"].unique()
    )
    print("Saved to:", OUTPUT_PATH)

    print("\nSamples per scenario:")
    print(dataset["scenario"].value_counts())

