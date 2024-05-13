from circuitree import CircuiTree
from circuitree.models import SimpleNetworkGrammar
from time import sleep
import numpy as np

grammar = SimpleNetworkGrammar(
    components=["A", "B", "C"],
    interactions=["activates", "inhibits"],
)


def get_bistability_reward(state, grammar, rg=None, expensive=False):
    """Returns a reward value for the given state (topology) based on
    whether it contains positive-feedback loops (PFLs). Assumes the
    state is a string in the format of SimpleNetworkGrammar."""

    # We list all types of PFLs with up to 3 components. Each three-letter
    # substring is an interaction in the circuit, and interactions are
    # separated by underscores.
    patterns = [
        "AAa",  # PAR - "AAa" means "A activates A"
        "ABi_BAi",  # Mutual inhibition - "A inhibits B, B inhibits A"
        "ABa_BAa",  # Mutual activation
        "ABa_BCa_CAa",  # Cycle of all activation
        "ABa_BCi_CAi",  # Cycle with two inhibitions
    ]

    # Mean reward increases with each PFL found (from 0.25 to 0.75)
    mean = 0.25
    for pattern in patterns:

        # The "has_pattern" method returns whether state contains the pattern.
        # It checks all possible renamings. For example, `has_pattern(s, 'AAa')`
        # checks whether the state `s` contains 'AAa', 'BBa', or 'CCa'.
        if grammar.has_pattern(state, pattern):
            mean += 0.1

    if expensive:  # Simulate a more expensive reward calculation
        sleep(0.1)

    # Use the default random number generator if none is provided
    rg = np.random.default_rng() if rg is None else rg

    return rg.normal(loc=mean, scale=0.1)


class BistabilityTree(CircuiTree):
    """A subclass of CircuiTree that searches for positive feedback networks.
    Uses the SimpleNetworkGrammar to encode network topologies. The grammar can
    be accessed with the `self.grammar` attribute."""

    def __init__(self, *args, **kwargs):
        kwargs = kwargs | {"grammar": grammar}
        super().__init__(*args, **kwargs)

    def get_reward(self, state: str, expensive: bool = False) -> float:
        """Returns a reward value for the given state (topology) based on
        whether it contains positive-feedback loops (PFLs)."""

        # `self.rg` is a Numpy random generator that can be seeded on initialization
        reward = get_bistability_reward(
            state, self.grammar, self.rg, expensive=expensive
        )
        return reward

    def get_mean_reward(self, state: str) -> float:
        """Returns the mean empirical reward value for the given state."""
        # The search graph is stored as a `networkx.DiGraph` in the `graph`
        # attribute. We can access the cumulative reward and # of visits for
        # each node (state) using the `reward` and `visits` attributes.
        r = self.graph.nodes[state].get("reward", 0)
        v = self.graph.nodes[state].get("visits", 1)
        return r / v

    def is_success(self, state: str) -> bool:
        """Returns whether a topology is a successful bistable circuit design."""
        if self.grammar.is_terminal(state):
            return self.get_mean_reward(state) > 0.5
        else:
            return False  # Ignore incomplete topologies
