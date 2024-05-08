### Must be run before you declare any multithreaded code using gevent, and it must be
### run as the first lines of the script! Patches the Python standard library to be
### compatible with gevent's cooperative threading.
from gevent import monkey

monkey.patch_all()

from celery import shared_task
from circuitree import CircuiTree
from circuitree.models import SimpleNetworkGrammar
import numpy as np
from time import sleep
from typing import Optional

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


class DistributedBistabilityTree(CircuiTree):
    """A subclass of CircuiTree that searches for positive feedback networks.
    Uses the SimpleNetworkGrammar to encode network topologies. In the
    SimpleNetworkGrammar, each topology or 'state' is specified using a 3-part
    string. For instance, a circuit with components 'A', 'B', and 'C' that
    repress each other in a cycle (i.e. the repressilator) would be represented
    as:

      *ABC::ABi_BCi_CAi

     - `::` separates circuit components from pairwise interactions
     - Components are uppercase letters, each type of interaction is a lowercase
       letter.
     - Pairwise interactions are 3-character strings. For exapmle, "ABi" means
       "A inhibits B"
     - A `*` at the beginning indicates that the state is terminal - the
       "termination" action was chosen, and the game has ended.

    The grammar can be accessed with the `self.grammar` attribute.

    This class is identical to BistabilityTree except reward evaluations are done in a
    distributed manner. Reward evaluations are dispatched to worker processes in the
    form of a Celery task. This allows other threads on the main node to continue
    their search iterations while the reward is being computed.

    Uses the `expensive` flag to simulate a more expensive reward calculation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(grammar=grammar, *args, **kwargs)

    def get_reward(self, state, expensive=False):
        # Generate a unique random seed and run the task in a worker
        seed = int(self.rg.integers(0, 2**32))
        result = get_reward_celery.delay(state, seed, expensive=expensive)
        reward = result.get()
        return reward


@shared_task
def run_mcts_parallel(
    n_steps: int,
    n_threads: int,
    expensive: bool = True,
    root="ABC::",
    run_kwargs: Optional[dict] = None,
    **kwargs
) -> tuple[dict, str]:
    """Runs a parallel MCTS search with the given number of steps and threads. Returns
    a dictionary of attributes (from `CircuiTree.to_dict()`) and the search graph object
    as a string in GML format. Together they can be used to reconstruct the search tree
    using `CircuiTree.from_file()`."""

    # Run the search in parallel
    run_kwargs = dict(expensive=expensive) | (run_kwargs or {})
    tree = DistributedBistabilityTree(root=root, **kwargs)
    tree.search_mcts_parallel(
        n_steps=n_steps, n_threads=n_threads, run_kwargs=run_kwargs
    )
    return tree.to_string()


def run_bistability_search_on_cloud(
    n_steps: int,
    n_threads: int,
    expensive: bool = True,
    root="ABC::",
    run_kwargs: Optional[dict] = None,
    **kwargs
) -> tuple[dict, str]:
    result = run_mcts_parallel.delay(
        n_steps=n_steps,
        n_threads=n_threads,
        expensive=expensive,
        root=root,
        run_kwargs=run_kwargs,
        **kwargs
    )
    serialized = result.get()
    tree = DistributedBistabilityTree.from_file(*serialized)
    return tree
