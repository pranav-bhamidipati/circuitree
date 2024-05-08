### Must be run before you declare any multithreaded code using gevent, and it must be
### run as the first lines of the script! Patches the Python standard library to be
### compatible with gevent's cooperative threading.
from gevent import monkey

monkey.patch_all()

from circuitree import CircuiTree
from circuitree.models import SimpleNetworkGrammar
from worker_tasks import get_reward_celery

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
        grammar = SimpleNetworkGrammar(
            components=["A", "B", "C"], interactions=["activates", "inhibits"]
        )
        super().__init__(grammar=grammar, *args, **kwargs)

    def get_reward(self, state, expensive=False):
        # Generate a unique random seed
        seed = int(self.rg.integers(0, 2**32))
        
        # Run the reward function in a worker process using Celery
        result = get_reward_celery.delay(state, seed, expensive=expensive)
        reward = result.get()
        return reward
