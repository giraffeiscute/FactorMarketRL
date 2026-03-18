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
    per_stock_alpha_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate alphas from either a shared group or fixed per-stock groups."""

    if per_stock_alpha_groups is not None:
        return pd.DataFrame(
            {
                "stock_id": list(stock_ids),
                "alpha": [
                    resolve_alpha_value(group_name, alpha_levels)
                    for group_name in per_stock_alpha_groups
                ],
            }
        )

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
