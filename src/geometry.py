import math

# Engineering assumptions
EARTH_RADIUS_KM = 6371.0
HAPS_ALTITUDE_KM = 20.0


def calculate_geometry(ground_distance_km):
    """
    Calculate HAPS-to-ground geometry using a spherical Earth.

    Input:
        ground_distance_km: Surface arc distance in km.

    Returns:
        Slant range in km and geometric elevation angle
        in degrees.
    """
    if not math.isfinite(ground_distance_km):
        raise ValueError("Ground distance must be finite.")

    if ground_distance_km < 0:
        raise ValueError("Ground distance cannot be negative.")

    R = EARTH_RADIUS_KM
    h = HAPS_ALTITUDE_KM

    # Convert surface distance to central angle (radians)
    phi = ground_distance_km / R

    # Straight-line distance between HAPS and receiver
    slant_range_km = math.sqrt(
        h**2 + 2 * R * (R + h) * (1 - math.cos(phi))
    )

    # Elevation angle measured above the receiver horizon
    vertical = (R + h) * math.cos(phi) - R
    horizontal = (R + h) * math.sin(phi)

    elevation_deg = math.degrees(
        math.atan2(vertical, horizontal)
    )

    return slant_range_km, elevation_deg


if __name__ == "__main__":
    # Simple numerical verification
    for distance in [0, 15, 50, 100]:
        slant, elevation = calculate_geometry(distance)

        print(
            f"Ground distance: {distance:6.1f} km | "
            f"Slant range: {slant:8.3f} km | "
            f"Elevation: {elevation:8.3f} deg"
        )