from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

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
    _default_per_stock_alpha_epsilon_groups,
    _default_per_stock_initial_prices,
    _default_per_stock_latent_state_params,
    build_default_config,
)
from toy_ff_generator.exposures import generate_exposures
from toy_ff_generator.factors import generate_factors
from toy_ff_generator.noise import generate_noise, resolve_epsilon_sigma
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
    validate_beta_df,
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


def _format_mu_vector(mu_vector: list[float] | tuple[float, ...]) -> str:
    return f"({float(mu_vector[0])},{float(mu_vector[1])},{float(mu_vector[2])})"


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
    dataset_number: int | None = None,
) -> str:
    state_name = _format_state_for_filename(state_sequence, market_state_setup)
    stock_count = int(simulation_setup["N"])
    time_count = int(simulation_setup["T"])
    if dataset_number is None:
        return f"{state_name}_{stock_count}_{time_count}_PL.csv"
    return f"{state_name}_{stock_count}_{time_count}_PL_{dataset_number}.csv"


def _build_market_index_csv_filename(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
    simulation_setup: Mapping[str, Any],
    dataset_number: int | None = None,
) -> str:
    state_name = _format_state_for_filename(state_sequence, market_state_setup)
    stock_count = int(simulation_setup["N"])
    time_count = int(simulation_setup["T"])
    if dataset_number is not None:
        return f"{state_name}_{stock_count}_{time_count}_market_index_{dataset_number}.csv"
    return f"{state_name}_{stock_count}_{time_count}_market_index.csv"


def _build_market_index_png_filename(
    state_sequence: list[int],
    market_state_setup: Mapping[str, Any],
    simulation_setup: Mapping[str, Any],
    dataset_number: int | None = None,
) -> str:
    state_name = _format_state_for_filename(state_sequence, market_state_setup)
    stock_count = int(simulation_setup["N"])
    time_count = int(simulation_setup["T"])
    if dataset_number is not None:
        return f"{state_name}_{stock_count}_{time_count}_market_index_{dataset_number}.png"
    return f"{state_name}_{stock_count}_{time_count}_market_index.png"


def _apply_overrides(
    config: dict[str, Any],
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
    dataset_count: int | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)

    if output_dir is not None:
        updated["output_setup"]["output_dir"] = output_dir
    if seed is not None:
        updated["simulation_setup"]["random_seed"] = seed
    if N is not None:
        updated["simulation_setup"]["N"] = N
        updated["latent_characteristic_setup"]["per_stock_params"] = _default_per_stock_latent_state_params(N)
        updated["alpha_epsilon_mode_setup"].update(_default_per_stock_alpha_epsilon_groups(N))
        updated["clipping_price_setup"]["per_stock_initial_price"] = _default_per_stock_initial_prices(N)
    if T is not None:
        updated["simulation_setup"]["T"] = T
    if dataset_count is not None:
        updated["simulation_setup"]["dataset_count"] = dataset_count
    if S is not None:
        updated["market_state_setup"]["state_sequence"] = [S] * updated["simulation_setup"]["T"]
        updated["market_state_setup"]["initial_state"] = S

    return updated


def _resolve_max_workers(
    batch_setup: Mapping[str, Any],
    resolved_dataset_count: int,
) -> int:
    configured_max_workers = batch_setup.get("max_workers")
    if configured_max_workers is None:
        resolved_max_workers = min(os.cpu_count() or 1, resolved_dataset_count)
    else:
        resolved_max_workers = int(configured_max_workers)

    return max(1, min(resolved_max_workers, resolved_dataset_count))


def _build_dataset_config(
    base_config: dict[str, Any],
    dataset_index: int,
) -> dict[str, Any]:
    dataset_config = deepcopy(base_config)
    dataset_config["simulation_setup"]["random_seed"] = (
        int(base_config["simulation_setup"]["random_seed"]) + dataset_index
    )
    return dataset_config


def _summarize_batch_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_number": result["dataset_number"],
        "run_seed": result["run_seed"],
        "output_paths": result["output_paths"],
    }


def _run_single_dataset_batch(
    config: dict[str, Any],
    dataset_number: int,
) -> dict[str, Any]:
    result = _run_simulation_from_config(
        config=config,
        dataset_number=dataset_number,
    )
    return _summarize_batch_result(result)


def _run_simulation_from_config(
    config: dict[str, Any],
    dataset_number: int | None = None,
) -> dict[str, Any]:
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
        mu_class_setup=config["mu_class_setup"],
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
        A=exposure_setup["A"],
        b=exposure_setup["b"],
    )
    validate_beta_df(
        beta_df=beta_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    alpha_df = generate_alpha(
        stock_ids=stock_ids,
        alpha_group=alpha_epsilon_mode_setup["alpha_group"],
        alpha_levels=alpha_epsilon_mode_setup["alpha_levels"],
        per_stock_alpha_groups=alpha_epsilon_mode_setup.get("per_stock_alpha_groups"),
    )

    epsilon_df = generate_noise(
        stock_ids=stock_ids,
        time_columns=time_columns,
        epsilon_group=alpha_epsilon_mode_setup["epsilon_group"],
        epsilon_levels=alpha_epsilon_mode_setup["epsilon_levels"],
        rng=rng,
        per_stock_epsilon_groups=alpha_epsilon_mode_setup.get("per_stock_epsilon_groups"),
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

    if latent_characteristic_setup["use_shared_latent_state_params"]:
        shared_mu = latent_characteristic_setup["shared_params"]["mu_Z"]
        mu_by_stock = {
            stock_id: _format_mu_vector(shared_mu)
            for stock_id in stock_ids
        }
    else:
        mu_by_stock = {
            stock_id: _format_mu_vector(mu_vector)
            for stock_id, mu_vector in zip(
                stock_ids,
                latent_characteristic_setup["per_stock_params"]["mu_i"],
                strict=True,
            )
        }
    alpha_group_by_stock = {
        stock_id: group_name
        for stock_id, group_name in zip(
            stock_ids,
            alpha_epsilon_mode_setup.get("per_stock_alpha_groups", []),
            strict=False,
        )
    }
    if not alpha_group_by_stock:
        alpha_group_by_stock = {
            stock_id: alpha_epsilon_mode_setup["alpha_group"]
            for stock_id in stock_ids
        }

    epsilon_group_by_stock = {
        stock_id: group_name
        for stock_id, group_name in zip(
            stock_ids,
            alpha_epsilon_mode_setup.get("per_stock_epsilon_groups", []),
            strict=False,
        )
    }
    if not epsilon_group_by_stock:
        epsilon_group_by_stock = {
            stock_id: alpha_epsilon_mode_setup["epsilon_group"]
            for stock_id in stock_ids
        }
    epsilon_variance_by_stock = {
        stock_id: resolve_epsilon_sigma(
            epsilon_group=group_name,
            epsilon_levels=alpha_epsilon_mode_setup["epsilon_levels"],
        )
        for stock_id, group_name in epsilon_group_by_stock.items()
    }
    panel_long_df["mu"] = panel_long_df["stock_id"].map(mu_by_stock)
    panel_long_df["alpha_group"] = panel_long_df["stock_id"].map(alpha_group_by_stock)
    panel_long_df["epsilon_group"] = panel_long_df["stock_id"].map(epsilon_group_by_stock)
    panel_long_df["epsilon_variance"] = panel_long_df["stock_id"].map(epsilon_variance_by_stock)

    panel_long_df = panel_long_df[
        [
            "stock_id",
            "t",
            "state",
            *FIRM_CHARACTERISTIC_COLUMNS,
            "mu",
            "alpha",
            "epsilon_variance",
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
        dataset_number=dataset_number,
    )
    market_index_csv_filename = _build_market_index_csv_filename(
        state_sequence=state_sequence,
        market_state_setup=market_state_setup,
        simulation_setup=simulation_setup,
        dataset_number=dataset_number,
    )
    market_index_png_filename = _build_market_index_png_filename(
        state_sequence=state_sequence,
        market_state_setup=market_state_setup,
        simulation_setup=simulation_setup,
        dataset_number=dataset_number,
    )
    dataset_label = (
        f" | dataset={dataset_number}"
        if dataset_number is not None
        else ""
    )

    output_paths = save_outputs(
        panel_long_df=panel_long_df,
        output_dir=output_setup["output_dir"],
        panel_filename=panel_filename,
        market_index_csv_filename=market_index_csv_filename,
        market_index_png_filename=market_index_png_filename,
        market_index_plot_title=(
            "market index | "
            f"state={_format_state_for_filename(state_sequence, market_state_setup)} | "
            f"N={int(simulation_setup['N'])} | "
            f"T={int(simulation_setup['T'])}"
            f"{dataset_label}"
        ),
        time_columns=time_columns,
    )

    return {
        "dataset_number": dataset_number,
        "run_seed": int(simulation_setup["random_seed"]),
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


def run_simulation(
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_config = _apply_overrides(
        config=build_default_config() if config is None else config,
        output_dir=output_dir,
        seed=seed,
        N=N,
        T=T,
        S=S,
    )
    return _run_simulation_from_config(config=resolved_config)


def run_batch_simulations(
    output_dir: str | None = None,
    seed: int | None = None,
    N: int | None = None,
    T: int | None = None,
    S: int | None = None,
    dataset_count: int | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_config = _apply_overrides(
        config=build_default_config() if config is None else config,
        output_dir=output_dir,
        seed=seed,
        N=N,
        T=T,
        S=S,
        dataset_count=dataset_count,
    )
    simulation_setup = base_config["simulation_setup"]
    batch_setup = base_config["batch_setup"]
    resolved_dataset_count = int(simulation_setup["dataset_count"])
    max_workers = _resolve_max_workers(
        batch_setup=batch_setup,
        resolved_dataset_count=resolved_dataset_count,
    )

    tasks = [
        (
            _build_dataset_config(base_config=base_config, dataset_index=dataset_index),
            dataset_index + 1,
        )
        for dataset_index in range(resolved_dataset_count)
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset_number = {
            executor.submit(_run_single_dataset_batch, dataset_config, dataset_number): dataset_number
            for dataset_config, dataset_number in tasks
        }

        completed_futures = as_completed(future_to_dataset_number)
        if tqdm is not None:
            completed_futures = tqdm(
                completed_futures,
                total=resolved_dataset_count,
                desc="Generating datasets",
                unit="dataset",
            )

        for future in completed_futures:
            results.append(future.result())

    return sorted(results, key=lambda item: int(item["dataset_number"]))


def main(batch: bool = True):
    if batch:
        return run_batch_simulations()
    return run_simulation()["panel_long_df"]


if __name__ == "__main__":
    main(batch=True)
