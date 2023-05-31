from typing import Iterable
import numpy as np

__all__ = ["regret"]


def regret(outcomes: Iterable[int | float], optimal_payout: int | float = -1):
    if optimal_payout < 0:
        optimal_payout = np.max(outcomes)
    return np.cumsum(optimal_payout - outcomes)
