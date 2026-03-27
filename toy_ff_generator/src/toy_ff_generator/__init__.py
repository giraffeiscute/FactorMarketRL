"""
這個模組是 `toy_ff_generator` 套件的進入點包裝。

它的用途是：
- 對外暴露 `run_simulation(...)` / `run_batch_simulations(...)`
- 避免在匯入套件時過早載入 `main.py`
- 讓 `python -m toy_ff_generator.main` 與一般套件匯入都能正常共存

主要輸入：
- `*args`, `**kwargs`：會原封不動轉交給 `main.run_simulation(...)`

主要輸出：
- 回傳 `main.run_simulation(...)` 的結果字典
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_simulation", "run_batch_simulations"]


def run_simulation(*args: Any, **kwargs: Any) -> Any:
    """延後載入 `main.py`，再轉呼叫主模擬函式。"""

    from toy_ff_generator.main import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)


def run_batch_simulations(*args: Any, **kwargs: Any) -> Any:
    """延後載入 `main.py`，再轉呼叫 batch 主模擬函式。"""

    from toy_ff_generator.main import run_batch_simulations as _run_batch_simulations

    return _run_batch_simulations(*args, **kwargs)
