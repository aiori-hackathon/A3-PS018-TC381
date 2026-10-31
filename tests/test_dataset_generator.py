import pandas as pd
import pytest

from src.dataset_generator import generate_dataset


@pytest.fixture(scope="module")
def dataset():
    # Generate once and reuse across tests
    return generate_dataset()


def test_dataset_not_empty(dataset):
    assert len(dataset) > 0


def test_eight_trajectories(dataset):
    assert dataset["trajectory_id"].nunique() == 8


def test_required_columns(dataset):
    required = [
        "trajectory_id",
        "ground_distance_km",
        "slant_range_km",
        "elevation_deg",
        "path_loss_db",
        "received_power_dbm",
        "snr_db",
    ]

    assert all(
        column in dataset.columns
        for column in required
    )


def test_no_missing_values(dataset):
    assert not dataset.isnull().any().any()


def test_reproducibility(dataset):
    # Compare with a fresh generation.
    # Cached propagation values avoid repeating P.528 work.
    second = generate_dataset()

    pd.testing.assert_frame_equal(dataset, second)