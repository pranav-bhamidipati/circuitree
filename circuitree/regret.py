from typing import Iterable, Optional
import numpy as np
from numba import vectorize

__all__ = ["regret"]


def regret(outcomes: Iterable[int | float], optimal_payout: int | float = -1):
    if optimal_payout < 0:
        optimal_payout = np.max(outcomes)
    return np.cumsum(optimal_payout - outcomes)


regret_vec = vectorize(regret)
