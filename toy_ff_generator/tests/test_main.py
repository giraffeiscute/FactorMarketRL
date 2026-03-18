"""Integration tests for the main pipeline."""

import json

import numpy as np
import pandas as pd

from toy_ff_generator.main import run_simulation
from toy_ff_generator.utils import build_firm_characteristics_excel_view


def test_main_pipeline_generates_panel_price_return_and_metadata_outputs(tmp_path) -> None:
    result = run_simulation(output_dir=str(tmp_path), seed=11, N=27, T=6, S=1)

    prices_path = tmp_path / "bull_27_6_price.csv"
    returns_path = tmp_path / "bull_27_6_return.csv"
    panel_path = tmp_path / "bull_27_6_panel_long.csv"
    metadata_path = tmp_path / "bull_27_6_metadata.json"

    assert prices_path.exists()
    assert returns_path.exists()
    assert panel_path.exists()
    assert metadata_path.exists()

    prices_df = pd.read_csv(prices_path, index_col=0)
    returns_df = pd.read_csv(returns_path, index_col=0)
    panel_df = pd.read_csv(panel_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert prices_df.shape == (27, 6)
    assert returns_df.shape == (27, 6)
    assert len(panel_df) == 162
    assert metadata["simulation_setup"]["N"] == 27
    assert metadata["simulation_setup"]["T"] == 6
    assert metadata["market_state_setup"]["resolved_state_sequence"] == [1, 1, 1, 1, 1, 1]
    assert panel_df["state"].tolist() == [1] * 162
    assert {"mu", "epsilon_variance"}.issubset(panel_df.columns)
    assert {"alpha_group", "epsilon_group"}.isdisjoint(panel_df.columns)
    assert {
        "characteristic_1",
        "characteristic_2",
        "characteristic_3",
    }.issubset(panel_df.columns)
    assert {
        "latent_characteristic_1_state",
        "latent_characteristic_2_state",
        "latent_characteristic_3_state",
    }.isdisjoint(panel_df.columns)

    latent_state_df = result["latent_state_df"].sort_values(["stock_id", "t"]).reset_index(drop=True)
    beta_df = result["beta_df"].sort_values(["stock_id", "t"]).reset_index(drop=True)
    exposure_matrix = np.asarray(result["config"]["exposure_setup"]["A"], dtype=float)
    intercept_vector = np.asarray(result["config"]["exposure_setup"]["b"], dtype=float)
    expected_beta_matrix = (
        latent_state_df[
            [
                "latent_characteristic_1_state",
                "latent_characteristic_2_state",
                "latent_characteristic_3_state",
            ]
        ].to_numpy(dtype=float)
        @ exposure_matrix.T
        + intercept_vector
    )
    assert np.allclose(
        beta_df[["beta_mkt", "beta_smb", "beta_hml"]].to_numpy(dtype=float),
        expected_beta_matrix,
    )
    assert panel_df.loc[panel_df["stock_id"] == "stock_000", "mu"].unique().tolist() == [
        "(-0.5,-0.5,-0.5)"
    ]
    assert panel_df["epsilon_variance"].round(10).tolist() == [0.02] * 162

    expected_return_wide = (
        result["panel_long_df"]
        .pivot(index="stock_id", columns="t", values="return")
        .sort_index()
        .reindex(columns=returns_df.columns)
    )
    expected_price_wide = (
        result["panel_long_df"]
        .pivot(index="stock_id", columns="t", values="price")
        .sort_index()
        .reindex(columns=prices_df.columns)
    )
    assert np.allclose(returns_df.to_numpy(dtype=float), expected_return_wide.to_numpy(dtype=float))
    assert np.allclose(prices_df.to_numpy(dtype=float), expected_price_wide.to_numpy(dtype=float))

    assert result["output_paths"]["prices"] == prices_path
    assert result["output_paths"]["returns"] == returns_path
    assert result["output_paths"]["panel_long"] == panel_path
    assert result["output_paths"]["metadata"] == metadata_path


def test_excel_view_uses_new_characteristic_axis_labels() -> None:
    firm_characteristics_df = pd.DataFrame(
        {
            "stock_id": ["stock_000", "stock_000"],
            "t": ["t_0", "t_1"],
            "characteristic_1": [1.5, 1.6],
            "characteristic_2": [0.9, 1.1],
            "characteristic_3": [-0.2, 0.3],
        }
    )

    excel_view = build_firm_characteristics_excel_view(
        firm_characteristics_df=firm_characteristics_df
    )

    assert excel_view.index.name == "firm_characteristic"
    assert excel_view.index.tolist() == [
        "characteristic_1",
        "characteristic_2",
        "characteristic_3",
    ]


def test_main_pipeline_covers_all_243_deterministic_stock_profiles(tmp_path) -> None:
    result = run_simulation(output_dir=str(tmp_path), seed=11, N=243, T=1, S=0)

    mu_by_stock = (
        result["panel_long_df"][["stock_id", "mu"]]
        .drop_duplicates()
        .sort_values("stock_id")
        .reset_index(drop=True)
    )
    alpha_groups = result["config"]["alpha_epsilon_mode_setup"]["per_stock_alpha_groups"]
    epsilon_groups = result["config"]["alpha_epsilon_mode_setup"]["per_stock_epsilon_groups"]
    profile_df = mu_by_stock.copy()
    profile_df["alpha_group"] = alpha_groups
    profile_df["epsilon_group"] = epsilon_groups
    profile_df = profile_df[["mu", "alpha_group", "epsilon_group"]].drop_duplicates()

    assert len(profile_df) == 243
    assert set(profile_df["alpha_group"]) == {"low", "mid", "high"}
    assert set(profile_df["epsilon_group"]) == {"low", "mid", "high"}
