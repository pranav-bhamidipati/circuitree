from celery import shared_task
from circuitree.models import SimpleNetworkGrammar
import numpy as np
from time import sleep

grammar = SimpleNetworkGrammar(
    components=["A", "B", "C"], interactions=["activates", "inhibits"]
)


def get_bistability_reward(
    state: str, rg: np.random.Generator, grammar: SimpleNetworkGrammar
) -> float:
    """Returns a reward value for the given state (topology) based on
    whether it contains positive-feedback loops (PFLs)."""

    # All types of PFLs with up to 3 components
    patterns = [
        "AAa",  # PAR - "AAa" means "A activates A"
        "ABi_BAi",  # Mutual inhibition - "A inhibits B, B inhibits A"
        "ABa_BAa",  # Mutual activation
        "ABa_BCa_CAa",  # Cycle of all activation
        "ABa_BCi_CAi",  # Cycle with two inhibitions
    ]

    # Mean reward increases from 0.25 to 0.75 based on the number of PFLS.
    mean = 0.25
    for pattern in patterns:

        ## The "has_pattern" method returns whether state contains the pattern.
        ## It checks all possible symmetries, so we only need to specify
        ## each pattern once (i.e. 'AAa' is equivalent to 'BBa' and 'CCa')
        if grammar.has_pattern(state, pattern):
            mean += 0.1

    # The CircuiTree object has its own random number generator
    return rg.normal(loc=mean, scale=0.1)


@shared_task
def get_reward_celery(state: str, seed: int, expensive: bool = False) -> float:
    """Returns a reward value for the given state based on how many types of positive
    feedback loops (PFLs) it contains. Same as `BistabilityTree.get_reward()`."""

    # Get a high-quality random seed and calculate the reward
    hq_seed = np.random.SeedSequence(seed).generate_state(1)[0]
    rg = np.random.default_rng(hq_seed)
    reward = get_bistability_reward(state, rg, grammar)

    if expensive:  # Simulate an expensive reward calculation
        sleep(0.1)

    return reward
