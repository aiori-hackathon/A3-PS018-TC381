import numpy as np

from src.geometry import calculate_geometry


def generate_trajectory(
    start_km=0.0,
    end_km=100.0,
    step_km=10.0
):
    """
    Generate receiver positions along a surface arc.

    All distances are in kilometres.
    """

    if not all(
        np.isfinite(value)
        for value in [start_km, end_km, step_km]
    ):
        raise ValueError("All inputs must be finite.")

    if start_km < 0:
        raise ValueError("Start distance cannot be negative.")

    if end_km < start_km:
        raise ValueError("End distance must be >= start distance.")

    if step_km <= 0:
        raise ValueError("Step must be positive.")

    # Include regularly spaced positions without
    # extending beyond the requested endpoint.
    count = int(
        np.floor((end_km - start_km) / step_km + 1e-10)
    ) + 1

    distances = start_km + np.arange(count) * step_km

    trajectory = []

    for distance in distances:
        slant, elevation = calculate_geometry(float(distance))

        trajectory.append({
            "ground_distance_km": float(distance),
            "slant_range_km": slant,
            "elevation_deg": elevation
        })

    return trajectory


if __name__ == "__main__":

    trajectory = generate_trajectory()

    print("HAPS Receiver Trajectory")
    print("-" * 65)

    for point in trajectory:
        print(
            f"Ground: {point['ground_distance_km']:6.1f} km | "
            f"Slant: {point['slant_range_km']:8.3f} km | "
            f"Elevation: {point['elevation_deg']:7.3f} deg"
        )

    print("-" * 65)
    print(f"Total positions: {len(trajectory)}")