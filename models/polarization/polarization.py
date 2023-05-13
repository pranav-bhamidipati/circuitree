# from itertools import product
from typing import Optional, Iterable, Mapping
from circuitree import SimpleNetworkTree


class PolarizationTree(SimpleNetworkTree):
    def __init__(
        self,
        winners: Optional[Iterable[str]] = None,
        win_probabilities: Optional[Mapping[str, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.winners = winners
        self.win_probabilities = win_probabilities
        if self.winners is not None:
            self.get_reward = lambda g: int(self.is_success(g))
            self.winners = set(self.winners)
        elif self.win_probabilities is not None:
            self.get_reward = self.draw_random_success
            self.win_probability = dict(win_probabilities)

    def is_success(self, genotype: str) -> bool:
        return self.get_unique_state(genotype).split("::")[-1] in self.winners

    def draw_random_success(self, genotype: str) -> int:
        unique_genotype = self.get_unique_state(genotype).split("::")[-1]
        p = self.win_probability.get(unique_genotype, 0.0)
        draws = self.rg.binomial(self.batch_size, p)
        return draws / self.batch_size
