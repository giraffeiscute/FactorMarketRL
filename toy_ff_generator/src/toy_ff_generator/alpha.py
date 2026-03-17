from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def resolve_alpha_value(
    alpha_group: str,
    alpha_levels: Mapping[str, float],
) -> float:
    """Return the configured alpha level for the selected group."""

    try:
        return float(alpha_levels[alpha_group])
    except KeyError as exc:
        raise ValueError(
            f"alpha_group must be one of {sorted(alpha_levels)}. Received {alpha_group!r}."
        ) from exc


def generate_alpha(
    stock_ids: Sequence[str],
    alpha_group: str,
    alpha_levels: Mapping[str, float],
) -> pd.DataFrame:
    """Generate a constant alpha for all stocks from the configured group."""

    alpha_value = resolve_alpha_value(
        alpha_group=alpha_group,
        alpha_levels=alpha_levels,
    )
    return pd.DataFrame(
        {
            "stock_id": list(stock_ids),
            "alpha": [alpha_value] * len(stock_ids),
        }
    )
