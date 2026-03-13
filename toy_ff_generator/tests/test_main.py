"""Integration tests for the main pipeline."""

import json

import pandas as pd

from toy_ff_generator.main import run_simulation


def test_main_pipeline_generates_required_outputs(tmp_path) -> None:
    result = run_simulation(output_dir=str(tmp_path), seed=11, N=4, T=6, S=1)

    returns_path = tmp_path / "returns.csv"
    prices_path = tmp_path / "prices.csv"
    panel_path = tmp_path / "panel_long.csv"
    metadata_path = tmp_path / "metadata.json"

    assert returns_path.exists()
    assert prices_path.exists()
    assert panel_path.exists()
    assert metadata_path.exists()

    returns_df = pd.read_csv(returns_path, index_col=0)
    prices_df = pd.read_csv(prices_path, index_col=0)
    panel_df = pd.read_csv(panel_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert returns_df.shape == (4, 6)
    assert prices_df.shape == (4, 6)
    assert len(panel_df) == 24
    assert metadata["simulation_setup"]["N"] == 4
    assert metadata["simulation_setup"]["T"] == 6
    assert metadata["market_state_setup"]["resolved_state_sequence"] == [1, 1, 1, 1, 1, 1]
    assert result["panel_long_df"].shape[0] == 24
