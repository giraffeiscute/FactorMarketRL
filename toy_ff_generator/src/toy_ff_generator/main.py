from __future__ import annotations

import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "toy_ff_generator_matplotlib"))

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False

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

STATUS_REFRESH_SECONDS = 0.5
NON_TTY_PRINT_INTERVAL_SECONDS = 5.0
TERMINAL_PHASE_WIDTH = 24
TERMINAL_FILE_WIDTH = 34
PHASE_SEQUENCE = [
    "building_state_sequence",
    "generating_factors",
    "generating_latent_states",
    "building_characteristics",
    "building_exposures",
    "generating_noise",
    "building_panel",
    "computing_returns",
    "generating_prices",
    "writing_outputs",
    "done",
]


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
        return f"{state_name}_{stock_count}_{time_count}_PL.parquet"
    return f"{state_name}_{stock_count}_{time_count}_PL_{dataset_number}.parquet"


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
        "status_path": result["status_path"],
        "batch_run_id": result["batch_run_id"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _run_single_dataset_batch(
    config: dict[str, Any],
    dataset_number: int,
    status_dir: str,
    batch_run_id: str,
) -> dict[str, Any]:
    status_path = _status_file_path(Path(status_dir), dataset_number)
    started_at = time.time()
    run_seed = int(config["simulation_setup"]["random_seed"])

    def report(
        *,
        status: str,
        phase: str,
        message: str,
        output_panel_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        _write_status_file(
            status_path,
            {
                "batch_run_id": batch_run_id,
                "dataset_number": int(dataset_number),
                "status": status,
                "phase": phase,
                "message": message,
                "started_at": _iso_timestamp(started_at),
                "updated_at": _iso_timestamp(),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "run_seed": run_seed,
                "pid": os.getpid(),
                "output_panel_path": output_panel_path or "",
                "error_message": error_message or "",
            },
        )

    report(
        status="STARTING",
        phase="building_state_sequence",
        message="Worker started.",
    )
    try:
        result = _run_simulation_from_config(
            config=config,
            dataset_number=dataset_number,
            status_reporter=report,
        )
    except Exception as exc:
        report(
            status="FAILED",
            phase="failed",
            message="Worker failed.",
            error_message="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        raise

    report(
        status="DONE",
        phase="done",
        message="Dataset finished.",
        output_panel_path=str(result["output_paths"]["panel_long"]),
    )
    result["status_path"] = str(status_path)
    result["batch_run_id"] = batch_run_id
    result["elapsed_seconds"] = round(time.time() - started_at, 3)
    return _summarize_batch_result(result)


def _run_simulation_from_config(
    config: dict[str, Any],
    dataset_number: int | None = None,
    status_reporter: Any | None = None,
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

    def report(phase: str, message: str, *, status: str = "RUNNING", output_panel_path: str | None = None) -> None:
        if status_reporter is not None:
            status_reporter(
                status=status,
                phase=phase,
                message=message,
                output_panel_path=output_panel_path,
            )

    report("building_state_sequence", "Resolving state sequence.")
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

    report("generating_factors", "Generating factor paths.")
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

    report("generating_latent_states", "Generating latent characteristic states.")
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

    report("building_characteristics", "Building firm characteristics.")
    firm_characteristics_df = state_to_firm_characteristics(latent_state_df=latent_state_df)
    validate_firm_characteristics_df(
        firm_characteristics_df=firm_characteristics_df,
        expected_rows=simulation_setup["N"] * simulation_setup["T"],
    )

    report("building_exposures", "Building exposures.")
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

    report("generating_noise", "Generating idiosyncratic noise.")
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

    report("building_panel", "Building panel dataframe.")
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

    report("computing_returns", "Computing returns and clipping.")
    panel_long_df = compute_raw_returns(panel_long_df)
    panel_long_df = clip_returns(
        panel_df=panel_long_df,
        limit_down=clipping_price_setup["limit_down"],
        limit_up=clipping_price_setup["limit_up"],
    )

    report("generating_prices", "Generating prices.")
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

    output_panel_path = str(Path(output_setup["output_dir"]) / panel_filename)
    report(
        "writing_outputs",
        "Writing parquet panel and market-index artifacts.",
        status="WRITING",
        output_panel_path=output_panel_path,
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
    output_path = Path(base_config["output_setup"]["output_dir"])
    status_dir = _resolve_status_dir(output_path)
    batch_run_id = _build_batch_run_id()
    status_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            _build_dataset_config(base_config=base_config, dataset_index=dataset_index),
            dataset_index + 1,
        )
        for dataset_index in range(resolved_dataset_count)
    ]

    results: list[dict[str, Any]] = []
    dashboard = _BatchDashboard(
        total=resolved_dataset_count,
        max_workers=max_workers,
        batch_run_id=batch_run_id,
        status_dir=status_dir,
    )
    future_to_dataset_number: dict[Any, int] = {}
    completed_results_by_dataset: dict[int, dict[str, Any]] = {}
    failed_errors_by_dataset: dict[int, str] = {}
    last_error_summary = ""
    started_at = time.time()
    dataset_numbers = [dataset_number for _, dataset_number in tasks]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for dataset_config, dataset_number in tasks:
            _write_status_file(
                _status_file_path(status_dir, dataset_number),
                {
                    "batch_run_id": batch_run_id,
                    "dataset_number": int(dataset_number),
                    "status": "QUEUED",
                    "phase": "queued",
                    "message": "Waiting for worker slot.",
                    "started_at": "",
                    "updated_at": _iso_timestamp(),
                    "elapsed_seconds": 0.0,
                    "run_seed": int(dataset_config["simulation_setup"]["random_seed"]),
                    "pid": None,
                    "output_panel_path": "",
                    "error_message": "",
                },
            )
            future = executor.submit(
                _run_single_dataset_batch,
                dataset_config,
                dataset_number,
                str(status_dir),
                batch_run_id,
            )
            future_to_dataset_number[future] = dataset_number

        dashboard.render(
            _load_status_snapshot(status_dir, dataset_numbers, batch_run_id=batch_run_id),
            started_at=started_at,
            last_error_summary=last_error_summary,
            pending_dataset_numbers=set(dataset_numbers),
            completed_results_by_dataset=completed_results_by_dataset,
            failed_errors_by_dataset=failed_errors_by_dataset,
        )
        pending = set(future_to_dataset_number)
        while pending:
            done, pending = wait(
                pending,
                timeout=STATUS_REFRESH_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                dataset_number = future_to_dataset_number[future]
                existing = _load_status_snapshot(
                    status_dir,
                    [dataset_number],
                    batch_run_id=batch_run_id,
                ).get(dataset_number, {})
                try:
                    result = future.result()
                    results.append(result)
                    completed_results_by_dataset[dataset_number] = result
                    _write_status_file(
                        _status_file_path(status_dir, dataset_number),
                        _merge_status_payload(
                            existing_payload=existing,
                            batch_run_id=batch_run_id,
                            dataset_number=dataset_number,
                            status="DONE",
                            phase="done",
                            message="Dataset finished (parent confirmed).",
                            elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
                            output_panel_path=str(result["output_paths"]["panel_long"]),
                            error_message="",
                        ),
                    )
                except Exception as exc:
                    last_error_summary = f"dataset {dataset_number} failed: {exc}"
                    failed_errors_by_dataset[dataset_number] = str(exc)
                    _write_status_file(
                        _status_file_path(status_dir, dataset_number),
                        _merge_status_payload(
                            existing_payload=existing,
                            batch_run_id=batch_run_id,
                            dataset_number=dataset_number,
                            status="FAILED",
                            phase="failed",
                            message="Worker failed before final status update.",
                            elapsed_seconds=float(existing.get("elapsed_seconds", 0.0) or 0.0),
                            output_panel_path=str(existing.get("output_panel_path", "")),
                            error_message=str(exc),
                        ),
                    )

            status_snapshot = _load_status_snapshot(
                status_dir,
                dataset_numbers,
                batch_run_id=batch_run_id,
            )
            dashboard.render(
                status_snapshot,
                started_at=started_at,
                last_error_summary=last_error_summary,
                pending_dataset_numbers={future_to_dataset_number[future] for future in pending},
                completed_results_by_dataset=completed_results_by_dataset,
                failed_errors_by_dataset=failed_errors_by_dataset,
            )

    final_snapshot = _load_status_snapshot(status_dir, dataset_numbers, batch_run_id=batch_run_id)
    dashboard.render(
        final_snapshot,
        started_at=started_at,
        last_error_summary=last_error_summary,
        final=True,
        pending_dataset_numbers=set(),
        completed_results_by_dataset=completed_results_by_dataset,
        failed_errors_by_dataset=failed_errors_by_dataset,
    )
    failed_datasets = sorted(failed_errors_by_dataset)
    if failed_datasets:
        raise RuntimeError(f"Dataset generation failed for datasets: {failed_datasets}")

    return sorted(results, key=lambda item: int(item["dataset_number"]))


def _status_file_path(status_dir: Path, dataset_number: int) -> Path:
    return status_dir / f"dataset_{int(dataset_number):03d}.json"


def _write_status_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _load_status_snapshot(
    status_dir: Path,
    dataset_numbers: list[int],
    *,
    batch_run_id: str | None = None,
) -> dict[int, dict[str, Any]]:
    snapshot: dict[int, dict[str, Any]] = {}
    for dataset_number in dataset_numbers:
        status_path = _status_file_path(status_dir, dataset_number)
        if not status_path.exists():
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if batch_run_id is not None and payload.get("batch_run_id") != batch_run_id:
            continue
        snapshot[dataset_number] = payload
    return snapshot


def _merge_status_payload(
    *,
    existing_payload: dict[str, Any],
    batch_run_id: str,
    dataset_number: int,
    status: str,
    phase: str,
    message: str,
    elapsed_seconds: float,
    output_panel_path: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        **existing_payload,
        "batch_run_id": batch_run_id,
        "dataset_number": int(dataset_number),
        "status": status,
        "phase": phase,
        "message": message,
        "started_at": existing_payload.get("started_at", ""),
        "updated_at": _iso_timestamp(),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "run_seed": existing_payload.get("run_seed"),
        "pid": existing_payload.get("pid"),
        "output_panel_path": output_panel_path,
        "error_message": error_message,
    }


def _resolve_status_dir(output_dir: Path) -> Path:
    return output_dir / "_status" / "toy_ff_generator"


def _build_batch_run_id() -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _iso_timestamp(timestamp: float | None = None) -> str:
    if timestamp is None:
        timestamp = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp))


def _parse_iso_timestamp(raw_value: str) -> float | None:
    if not raw_value:
        return None
    try:
        return time.mktime(time.strptime(raw_value, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value.ljust(width)
    if width <= 1:
        return value[:width]
    return f"{value[: width - 1]}~"


class _BatchDashboard:
    def __init__(
        self,
        *,
        total: int,
        max_workers: int,
        batch_run_id: str,
        status_dir: Path,
    ) -> None:
        self.total = total
        self.max_workers = max_workers
        self.batch_run_id = batch_run_id
        self.status_dir = status_dir
        self.is_tty = sys.stdout.isatty()
        self.use_rich_live = bool(HAS_RICH and self.is_tty)
        self._last_render = ""
        self._last_non_tty_print = 0.0
        self._console = Console(file=sys.stdout) if self.use_rich_live else None
        self._live: Live | None = None

    def render(
        self,
        status_snapshot: dict[int, dict[str, Any]],
        *,
        started_at: float,
        last_error_summary: str,
        pending_dataset_numbers: set[int],
        completed_results_by_dataset: dict[int, dict[str, Any]],
        failed_errors_by_dataset: dict[int, str],
        final: bool = False,
    ) -> None:
        lines = self._build_lines(
            status_snapshot=status_snapshot,
            started_at=started_at,
            last_error_summary=last_error_summary,
            pending_dataset_numbers=pending_dataset_numbers,
            completed_results_by_dataset=completed_results_by_dataset,
            failed_errors_by_dataset=failed_errors_by_dataset,
            final=final,
        )

        if self.use_rich_live:
            signature = "\n".join(lines)
            if signature != self._last_render or final:
                renderable = self._build_rich_renderable(
                    lines=lines,
                    status_snapshot=status_snapshot,
                    pending_dataset_numbers=pending_dataset_numbers,
                    completed_results_by_dataset=completed_results_by_dataset,
                    failed_errors_by_dataset=failed_errors_by_dataset,
                    started_at=started_at,
                    last_error_summary=last_error_summary,
                    final=final,
                )
                if self._live is None:
                    self._live = Live(
                        renderable,
                        console=self._console,
                        refresh_per_second=max(1, int(round(1.0 / STATUS_REFRESH_SECONDS))),
                        transient=False,
                        auto_refresh=False,
                    )
                    self._live.start()
                else:
                    self._live.update(renderable, refresh=False)
                self._live.refresh()
                self._last_render = signature
            if final and self._live is not None:
                self._live.stop()
                self._live = None
            return

        output = "\n".join(lines)
        now = time.time()
        if final or output != self._last_render and now - self._last_non_tty_print >= NON_TTY_PRINT_INTERVAL_SECONDS:
            print(output, flush=True)
            self._last_render = output
            self._last_non_tty_print = now

    def _build_lines(
        self,
        *,
        status_snapshot: dict[int, dict[str, Any]],
        started_at: float,
        last_error_summary: str,
        pending_dataset_numbers: set[int],
        completed_results_by_dataset: dict[int, dict[str, Any]],
        failed_errors_by_dataset: dict[int, str],
        final: bool,
    ) -> list[str]:
        now = time.time()
        completed = len(completed_results_by_dataset)
        failed = len(failed_errors_by_dataset)
        running_payloads = [
            payload
            for dataset_number, payload in status_snapshot.items()
            if dataset_number in pending_dataset_numbers
            and payload.get("status") in {"STARTING", "RUNNING", "WRITING"}
        ]
        queued = max(0, self.total - completed - failed - len(running_payloads))
        elapsed = now - started_at
        est_remaining = None
        if completed > 0:
            est_remaining = max(0.0, elapsed / float(completed) * float(self.total - completed))

        header = (
            f"toy_ff_generator batch {self.batch_run_id} | status_dir={self.status_dir} | "
            f"workers={self.max_workers}"
        )
        summary = (
            f"completed={completed}/{self.total} running={len(running_payloads)} "
            f"failed={failed} queued={queued} elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(est_remaining)}"
        )
        lines = [header, summary, ""]
        lines.extend(
            self._build_table_lines(
                running_payloads,
                status_snapshot,
                pending_dataset_numbers=pending_dataset_numbers,
                now=now,
            )
        )
        if last_error_summary:
            lines.extend(["", f"last_error: {last_error_summary}"])
        if final:
            lines.extend(["", *self._build_final_summary(status_snapshot)])
        return lines

    def _build_rich_renderable(
        self,
        *,
        lines: list[str],
        status_snapshot: dict[int, dict[str, Any]],
        pending_dataset_numbers: set[int],
        completed_results_by_dataset: dict[int, dict[str, Any]],
        failed_errors_by_dataset: dict[int, str],
        started_at: float,
        last_error_summary: str,
        final: bool,
    ) -> Any:
        del lines, started_at
        now = time.time()
        completed = len(completed_results_by_dataset)
        failed = len(failed_errors_by_dataset)
        running_payloads = [
            payload
            for dataset_number, payload in status_snapshot.items()
            if dataset_number in pending_dataset_numbers
            and payload.get("status") in {"STARTING", "RUNNING", "WRITING"}
        ]
        queued = max(0, self.total - completed - failed - len(running_payloads))

        summary_table = Table.grid(expand=True)
        summary_table.add_column(justify="left")
        summary_table.add_column(justify="left")
        summary_table.add_row(
            f"[bold]Run[/bold] {self.batch_run_id}",
            f"[bold]Workers[/bold] {self.max_workers}",
        )
        summary_table.add_row(
            f"[bold]Completed[/bold] {completed}/{self.total}",
            f"[bold]Running[/bold] {len(running_payloads)}",
        )
        summary_table.add_row(
            f"[bold]Failed[/bold] {failed}",
            f"[bold]Queued[/bold] {queued}",
        )
        if last_error_summary:
            summary_table.add_row(
                "[bold red]Last Error[/bold red]",
                last_error_summary,
            )

        worker_table = Table(
            title="Dataset Workers",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        worker_table.add_column("Dataset", justify="right", no_wrap=True)
        worker_table.add_column("Status", no_wrap=True)
        worker_table.add_column("Phase")
        worker_table.add_column("Elapsed", justify="right", no_wrap=True)
        worker_table.add_column("Output")

        active_rows = sorted(
            running_payloads,
            key=lambda payload: int(payload.get("dataset_number", 0)),
        )
        if not active_rows:
            active_rows = sorted(
                (
                    payload
                    for dataset_number, payload in status_snapshot.items()
                    if dataset_number in pending_dataset_numbers
                ),
                key=lambda payload: int(payload.get("dataset_number", 0)),
            )[: min(5, max(1, len(pending_dataset_numbers)))]

        if not active_rows and final:
            active_rows = sorted(
                (
                    payload
                    for payload in status_snapshot.values()
                    if payload.get("status") in {"DONE", "FAILED"}
                ),
                key=lambda payload: int(payload.get("dataset_number", 0)),
            )[: min(5, len(status_snapshot))]

        if active_rows:
            for payload in active_rows[: max(self.max_workers, 5)]:
                status_value = str(payload.get("status", "-"))
                status_style = {
                    "DONE": "[green]DONE[/green]",
                    "FAILED": "[red]FAILED[/red]",
                    "WRITING": "[yellow]WRITING[/yellow]",
                    "RUNNING": "[blue]RUNNING[/blue]",
                    "STARTING": "[cyan]STARTING[/cyan]",
                    "QUEUED": "[white]QUEUED[/white]",
                }.get(status_value, status_value)
                output_name = Path(str(payload.get("output_panel_path") or "-")).name
                worker_table.add_row(
                    str(int(payload.get("dataset_number", 0))),
                    status_style,
                    _truncate(str(payload.get("phase", "-")), TERMINAL_PHASE_WIDTH),
                    _format_duration(self._effective_elapsed_seconds(payload, now)),
                    _truncate(output_name, TERMINAL_FILE_WIDTH),
                )
        else:
            worker_table.add_row("-", "-", "No active workers", "-", "-")

        renderables: list[Any] = [
            Panel(summary_table, title="toy_ff_generator", border_style="blue"),
            worker_table,
        ]
        if final:
            final_table = Table(
                title="Final Summary",
                show_header=True,
                header_style="bold magenta",
                expand=True,
            )
            final_table.add_column("Section", style="bold")
            final_table.add_column("Value")
            slowest_lines = self._build_final_summary(status_snapshot)
            current_section = ""
            for line in slowest_lines:
                if line.endswith(":"):
                    current_section = line[:-1]
                    continue
                final_table.add_row(current_section or "summary", line.strip())
            renderables.append(final_table)
        return Group(*renderables)

    def _build_table_lines(
        self,
        running_payloads: list[dict[str, Any]],
        status_snapshot: dict[int, dict[str, Any]],
        *,
        pending_dataset_numbers: set[int],
        now: float,
    ) -> list[str]:
        active_rows = sorted(
            running_payloads,
            key=lambda payload: int(payload.get("dataset_number", 0)),
        )
        if not active_rows:
            recent_rows = sorted(
                (
                    payload
                    for dataset_number, payload in status_snapshot.items()
                    if dataset_number in pending_dataset_numbers
                ),
                key=lambda payload: int(payload.get("dataset_number", 0)),
            )[: min(5, max(1, len(pending_dataset_numbers)))]
            active_rows = recent_rows

        lines = [
            "dataset  status    phase                    elapsed  output",
            "-------  --------  ------------------------  -------  ----------------------------------",
        ]
        for payload in active_rows[: max(self.max_workers, 5)]:
            output_name = Path(str(payload.get("output_panel_path") or "-")).name
            elapsed_seconds = self._effective_elapsed_seconds(payload, now)
            lines.append(
                f"{int(payload.get('dataset_number', 0)):>7d}  "
                f"{_truncate(str(payload.get('status', '-')), 8)}  "
                f"{_truncate(str(payload.get('phase', '-')), TERMINAL_PHASE_WIDTH)}  "
                f"{_format_duration(elapsed_seconds):>7}  "
                f"{_truncate(output_name, TERMINAL_FILE_WIDTH)}"
            )
        return lines

    def _effective_elapsed_seconds(self, payload: dict[str, Any], now: float) -> float:
        status = str(payload.get("status", ""))
        if status in {"STARTING", "RUNNING", "WRITING"}:
            started_at = str(payload.get("started_at", "")).strip()
            started_ts = _parse_iso_timestamp(started_at)
            if started_ts is not None:
                return max(0.0, now - started_ts)
        return float(payload.get("elapsed_seconds", 0.0))

    def _build_final_summary(self, status_snapshot: dict[int, dict[str, Any]]) -> list[str]:
        done_payloads = [
            payload for payload in status_snapshot.values() if payload.get("status") == "DONE"
        ]
        slowest = sorted(
            done_payloads,
            key=lambda payload: float(payload.get("elapsed_seconds", 0.0)),
            reverse=True,
        )[:5]
        failed = sorted(
            (
                payload for payload in status_snapshot.values() if payload.get("status") == "FAILED"
            ),
            key=lambda payload: int(payload.get("dataset_number", 0)),
        )
        lines = ["slowest_datasets:"]
        if not slowest:
            lines.append("  none")
        else:
            for payload in slowest:
                lines.append(
                    f"  dataset {int(payload['dataset_number'])}: "
                    f"{_format_duration(float(payload.get('elapsed_seconds', 0.0)))} "
                    f"phase={payload.get('phase', '-')}"
                )
        lines.append("failed_datasets:")
        if not failed:
            lines.append("  none")
        else:
            for payload in failed:
                error_message = str(payload.get("error_message", "")).splitlines()[:1]
                summary = error_message[0] if error_message else "unknown error"
                lines.append(f"  dataset {int(payload['dataset_number'])}: {summary}")
        return lines


def main(batch: bool = True):
    if batch:
        return run_batch_simulations()
    return run_simulation()["panel_long_df"]


if __name__ == "__main__":
    main(batch=True)
