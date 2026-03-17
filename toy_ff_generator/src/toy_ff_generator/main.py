from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

from toy_ff_generator.alpha import generate_alpha
from toy_ff_generator.characteristics import (
    FIRM_CHARACTERISTIC_COLUMNS,
    generate_latent_characteristic_states,
    state_to_firm_characteristics,
)
from toy_ff_generator.config import (
    STATE_NAME_MAP,
    STATE_ORDER,
    _default_per_stock_initial_prices,
    _default_per_stock_latent_state_params,
    build_default_config,
)
from toy_ff_generator.exposures import generate_exposures
from toy_ff_generator.factors import generate_factors
from toy_ff_generator.noise import generate_noise
from toy_ff_generator.returns import (
    build_panel,
    clip_returns,
    compute_raw_returns,
    generate_prices,
)
from toy_ff_generator.utils import (
    make_stock_ids,
    make_time_columns,
    save_outputs,
    set_random_seed,
)
from toy_ff_generator.validation import (
    validate_component_row_count,
    validate_firm_characteristics_df,
    validate_latent_state_df,
    validate_panel_row_count,
    validate_simulation_inputs,
)


def _build_state_sequence(
    t_count: int,
    market_state_setup: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[int]:
    manual_sequence = market_state_setup.get("state_sequence")
    if manual_sequence is not None:
        return [int(state) for state in manual_sequence]

    transition_matrix = np.asarray(market_state_setup["transition_matrix"], dtype=float)
    current_state = int(market_state_setup["initial_state"])
    state_sequence = [current_state]

    for _ in range(1, t_count):
        current_index = STATE_ORDER.index(current_state)
        current_state = int(rng.choice(STATE_ORDER, p=transition_matrix[current_index]))
        state_sequence.append(current_state)

    return state_sequence


def _build_initial_prices(
    stock_ids: list[str],
    clipping_price_setup: Mapping[str, Any],
) -> dict[str, float]:
    if clipping_price_setup["shared_init_price"]:
        initial_price = float(clipping_price_setup["initial_price"])
        return {stock_id: initial_price for stock_id in stock_ids}

    per_stock_initial_price = clipping_price_setup["per_stock_initial_price"]
    return {
        stock_id: float(price)
        for stock_id, price in zip(stock_ids, per_stock_initial_price, strict=True)
    }


def _format_state_for_filename(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
) -> str:
    unique_states = sorted(set(int(state) for state in state_sequence))
    if len(unique_states) == 1:
        return STATE_NAME_MAP.get(unique_states[0], str(unique_states[0]))

    if market_state_setup.get("state_sequence") is not None:
        return "sequence"

    return "markov"


def _build_panel_filename(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
    simulation_setup: Mapping[str, Any],
) -> str:
    state_name = _format_state_for_filename(state_sequence, market_state_setup)
    stock_count = int(simulation_setup["N"])
    time_count = int(simulation_setup["T"])
    return f"{state_name}_{stock_count}_{time_count}_panel_long.csv"


def _build_price_filename(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
    simulation_setup: Mapping[str, Any],
) -> str:
    state_name = _format_state_for_filename(state_sequence, market_state_setup)
    stock_count = int(simulation_setup["N"])
    time_count = int(simulation_setup["T"])
    return f"{state_name}_{stock_count}_{time_count}_price.csv"


def _build_metadata_filename(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
    simulation_setup: Mapping[str, Any],
) -> str:
    state_name = _format_state_for_filename(state_sequence, market_state_setup)
    stock_count = int(simulation_setup["N"])
    time_count = int(simulation_setup["T"])
    return f"{state_name}_{stock_count}_{time_count}_metadata.json"


def _apply_overrides(
    config: dict[str, Any],
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)

    if output_dir is not None:
        updated["output_setup"]["output_dir"] = output_dir
    if seed is not None:
        updated["simulation_setup"]["random_seed"] = seed
    if N is not None:
        updated["simulation_setup"]["N"] = N
        updated["latent_characteristic_setup"]["per_stock_params"] = _default_per_stock_latent_state_params(N)
        updated["clipping_price_setup"]["per_stock_initial_price"] = _default_per_stock_initial_prices(N)
    if T is not None:
        updated["simulation_setup"]["T"] = T
    if S is not None:
        updated["market_state_setup"]["state_sequence"] = [S] * updated["simulation_setup"]["T"]
        updated["market_state_setup"]["initial_state"] = S

    return updated


def run_simulation(
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
) -> dict[str, Any]:
    config = _apply_overrides(
        config=build_default_config(),
        output_dir=output_dir,
        seed=seed,
        N=N,
        T=T,
        S=S,
    )

    simulation_setup = config["simulation_setup"]
    market_state_setup = config["market_state_setup"]
    factor_vector_ar_setup = config["factor_vector_ar_setup"]
    latent_characteristic_setup = config["latent_characteristic_setup"]
    exposure_setup = config["exposure_setup"]
    alpha_epsilon_mode_setup = config["alpha_epsilon_mode_setup"]
    clipping_price_setup = config["clipping_price_setup"]
    output_setup = config["output_setup"]

    stock_ids = make_stock_ids(simulation_setup["N"])
    time_columns = make_time_columns(simulation_setup["T"])
    rng = set_random_seed(simulation_setup["random_seed"])

    state_sequence = _build_state_sequence(
        t_count=simulation_setup["T"],
        market_state_setup=market_state_setup,
        rng=rng,
    )
    config["market_state_setup"]["resolved_state_sequence"] = state_sequence

    validate_simulation_inputs(
        N=simulation_setup["N"],
        T=simulation_setup["T"],
        market_state_setup={**market_state_setup, "state_sequence": state_sequence},
        factor_vector_ar_setup=factor_vector_ar_setup,
        latent_characteristic_setup=latent_characteristic_setup,
        exposure_setup=exposure_setup,
        alpha_epsilon_mode_setup=alpha_epsilon_mode_setup,
        clipping_price_setup=clipping_price_setup,
    )

    factor_df = generate_factors(
        t_count=simulation_setup["T"],
        state_sequence=state_sequence,
        X0=factor_vector_ar_setup["X0"],
        Phi=factor_vector_ar_setup["Phi"],
        Delta=factor_vector_ar_setup.get("Delta"),
        Sigma_X_bear=factor_vector_ar_setup["Sigma_X_bear"],
        Sigma_X_neutral=factor_vector_ar_setup["Sigma_X_neutral"],
        Sigma_X_bull=factor_vector_ar_setup["Sigma_X_bull"],
        rng=rng,
        mu_bear=factor_vector_ar_setup.get("mu_bear"),
        mu_neutral=factor_vector_ar_setup.get("mu_neutral"),
        mu_bull=factor_vector_ar_setup.get("mu_bull"),
    )

    latent_state_df = generate_latent_characteristic_states(
        stock_ids=stock_ids,
        time_columns=time_columns,
        state_sequence=state_sequence,
        use_shared_latent_state_params=latent_characteristic_setup[
            "use_shared_latent_state_params"
        ],
        shared_params=latent_characteristic_setup["shared_params"],
        per_stock_params=latent_characteristic_setup["per_stock_params"],
        rng=rng,
    )
    validate_latent_state_df(
        latent_state_df=latent_state_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    firm_characteristics_df = state_to_firm_characteristics(latent_state_df=latent_state_df)
    validate_firm_characteristics_df(
        firm_characteristics_df=firm_characteristics_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    beta_df = generate_exposures(
        latent_state_df=latent_state_df,
        a_mkt=exposure_setup["a_mkt"],
        b_mkt=exposure_setup["b_mkt"],
        a_smb=exposure_setup["a_smb"],
        b_smb=exposure_setup["b_smb"],
        a_hml=exposure_setup["a_hml"],
        b_hml=exposure_setup["b_hml"],
    )
    validate_component_row_count(
        name="beta_df",
        df=beta_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    alpha_df = generate_alpha(
        stock_ids=stock_ids,
        alpha_group=alpha_epsilon_mode_setup["alpha_group"],
        alpha_levels=alpha_epsilon_mode_setup["alpha_levels"],
    )

    epsilon_df = generate_noise(
        stock_ids=stock_ids,
        time_columns=time_columns,
        epsilon_group=alpha_epsilon_mode_setup["epsilon_group"],
        epsilon_levels=alpha_epsilon_mode_setup["epsilon_levels"],
        rng=rng,
    )
    validate_component_row_count(
        name="epsilon_df",
        df=epsilon_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    panel_long_df = build_panel(
        firm_characteristics_df=firm_characteristics_df,
        beta_df=beta_df,
        alpha_df=alpha_df,
        epsilon_df=epsilon_df,
        factor_df=factor_df,
    )
    validate_panel_row_count(
        panel_df=panel_long_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    panel_long_df = compute_raw_returns(panel_long_df)
    panel_long_df = clip_returns(
        panel_df=panel_long_df,
        limit_down=clipping_price_setup["limit_down"],
        limit_up=clipping_price_setup["limit_up"],
    )

    initial_prices = _build_initial_prices(
        stock_ids=stock_ids,
        clipping_price_setup=clipping_price_setup,
    )
    panel_long_df = generate_prices(
        panel_df=panel_long_df,
        initial_prices=initial_prices,
        time_columns=time_columns,
    )

    panel_long_df = panel_long_df[
        [
            "stock_id",
            "t",
            "state",
            *FIRM_CHARACTERISTIC_COLUMNS,
            "alpha",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
            "MKT",
            "SMB",
            "HML",
            "epsilon",
            "raw_return",
            "return",
            "price",
        ]
    ].copy()

    panel_filename = _build_panel_filename(
        state_sequence=state_sequence,
        market_state_setup=market_state_setup,
        simulation_setup=simulation_setup,
    )
    price_filename = _build_price_filename(
        state_sequence=state_sequence,
        market_state_setup=market_state_setup,
        simulation_setup=simulation_setup,
    )
    metadata_filename = _build_metadata_filename(
        state_sequence=state_sequence,
        market_state_setup=market_state_setup,
        simulation_setup=simulation_setup,
    )

    output_paths = save_outputs(
        panel_long_df=panel_long_df,
        firm_characteristics_df=firm_characteristics_df,
        output_dir=output_setup["output_dir"],
        panel_filename=panel_filename,
        price_filename=price_filename,
        metadata_filename=metadata_filename,
        time_columns=time_columns,
        metadata=config,
    )

    return {
        "config": config,
        "state_sequence": state_sequence,
        "factor_df": factor_df,
        "latent_state_df": latent_state_df,
        "firm_characteristics_df": firm_characteristics_df,
        "beta_df": beta_df,
        "alpha_df": alpha_df,
        "epsilon_df": epsilon_df,
        "panel_long_df": panel_long_df,
        "output_paths": output_paths,
    }


def main() -> pd.DataFrame:
    result = run_simulation()
    return result["panel_long_df"]


if __name__ == "__main__":
    main()
