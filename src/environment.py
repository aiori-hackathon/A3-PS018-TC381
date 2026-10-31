import math
from itur.models.itu838 import rain_specific_attenuation

RAINFALL_RATE_MM_H = 25.0
RAIN_LAYER_HEIGHT_KM = 5.0


def calculate_rain_attenuation(
    frequency_ghz,
    elevation_deg,
    rainfall_rate_mm_h=RAINFALL_RATE_MM_H
):
    if elevation_deg <= 0:
        return 0.0

    if rainfall_rate_mm_h < 0:
        raise ValueError("Rainfall rate cannot be negative.")

    if frequency_ghz <= 0:
        raise ValueError("Frequency must be positive.")

    if RAIN_LAYER_HEIGHT_KM <= 0:
        raise ValueError("Rain layer height must be positive.")

    specific_attenuation = rain_specific_attenuation(
        rainfall_rate_mm_h,
        frequency_ghz,
        elevation_deg,
        90.0
    ).value

    elevation_rad = math.radians(elevation_deg)

    effective_path_km = min(
        RAIN_LAYER_HEIGHT_KM / math.sin(elevation_rad),
        10.0
    )

    return float(specific_attenuation) * effective_path_km


if __name__ == "__main__":
    result = calculate_rain_attenuation(2.0, 30.0)
    print("Rain attenuation:", result, "dB")