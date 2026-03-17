from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def resolve_epsilon_sigma(
    epsilon_group: str,
    epsilon_levels: Mapping[str, float],
) -> float:
    """Return the configured epsilon sigma for the selected group."""

    try:
        return float(epsilon_levels[epsilon_group])
    except KeyError as exc:
        raise ValueError(
            f"epsilon_group must be one of {sorted(epsilon_levels)}. Received {epsilon_group!r}."
        ) from exc


def generate_noise(
    stock_ids: Sequence[str],
    time_columns: Sequence[str],
    epsilon_group: str,
    epsilon_levels: Mapping[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate epsilon draws with a shared sigma selected from config."""

    sigma_epsilon = resolve_epsilon_sigma(
        epsilon_group=epsilon_group,
        epsilon_levels=epsilon_levels,
    )

    rows: list[dict[str, float | str]] = []
    for stock_id in stock_ids:
        draws = rng.normal(loc=0.0, scale=sigma_epsilon, size=len(time_columns))
        for time_label, epsilon in zip(time_columns, draws, strict=True):
            rows.append(
                {
                    "stock_id": stock_id,
                    "t": time_label,
                    "epsilon": float(epsilon),
                }
            )

    return pd.DataFrame(rows)
