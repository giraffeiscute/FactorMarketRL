"""Integration tests for the main pipeline."""

import json

import numpy as np
import pandas as pd

from toy_ff_generator.main import build_default_config, run_simulation
from toy_ff_generator.utils import build_firm_characteristics_excel_view


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
    assert "state" in panel_df.columns
    assert panel_df["state"].tolist() == [1] * 24
    assert {"firm_size", "book_to_price"}.issubset(panel_df.columns)
    assert {"latent_size_state", "latent_book_to_price_state"}.isdisjoint(panel_df.columns)
    assert np.all(panel_df[["firm_size", "book_to_price"]].to_numpy(dtype=float) > 0.0)
    assert result["panel_long_df"].shape[0] == 24
    assert result["factor_df"]["state"].tolist() == [1] * 6

    latent_state_df = result["latent_state_df"].sort_values(["stock_id", "t"]).reset_index(drop=True)
    beta_df = result["beta_df"].sort_values(["stock_id", "t"]).reset_index(drop=True)
    expected_beta_mkt = 1.0 + 0.03 * (
        latent_state_df["latent_size_state"] + latent_state_df["latent_book_to_price_state"]
    )
    expected_beta_smb = -latent_state_df["latent_size_state"]
    expected_beta_hml = latent_state_df["latent_book_to_price_state"]

    assert np.allclose(beta_df["beta_mkt"], expected_beta_mkt)
    assert np.allclose(beta_df["beta_smb"], expected_beta_smb)
    assert np.allclose(beta_df["beta_hml"], expected_beta_hml)

    excel_path = result["output_paths"]["excel_workbook"]
    assert excel_path is None or excel_path.exists()


def test_default_latent_state_mu_i_is_per_stock_descending() -> None:
    config = build_default_config()
    latent_characteristic_setup = config["latent_characteristic_setup"]
    mu_i = latent_characteristic_setup["per_stock_params"]["mu_i"]

    assert latent_characteristic_setup["use_shared_latent_state_params"] is False
    assert np.allclose(
        mu_i,
        [
            [0.08, 0.05],
            [0.065, 0.04],
            [0.05, 0.03],
            [0.035, 0.02],
            [0.02, 0.01],
        ],
    )


def test_excel_view_uses_firm_characteristics_as_row_labels() -> None:
    firm_characteristics_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_000"],
            "t": ["t_0", "t_1"],
            "firm_size": [1.5, 1.6],
            "book_to_price": [0.9, 1.1],
        }
    )

    excel_view = build_firm_characteristics_excel_view(
        firm_characteristics_df=firm_characteristics_df
    )

    assert excel_view.index.name == "firm_characteristic"
    assert excel_view.index.tolist() == ["firm_size", "book_to_price"]
