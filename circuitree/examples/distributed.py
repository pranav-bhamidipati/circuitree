### Must be run before you declare any multithreaded code using gevent, and it must be
### run as the first lines of the script! Patches the Python standard library to be
### compatible with gevent's cooperative threading.
from gevent import monkey

monkey.patch_all()

from circuitree.examples.sequential import BistabilityTree
from circuitree.examples.worker_app import app, get_reward_celery


class DistributedBistabilityTree(BistabilityTree):
    """A subclass of BistabilityTree with parallel reward evaluations. This class is
    identical to BistabilityTree, except that it uses a Celery task to compute the
    reward function in a separate process. This allows the main thread to continue
    sampling the search space while the reward is being computed.

    Uses the `expensive` flag to simulate a more expensive reward calculation.
    """

    def get_reward(self, state, expensive=False):
        # Generate a unique random seed and run the task in a worker
        seed = int(self.rg.integers(0, 2**32))
        result = get_reward_celery.delay(state, seed, expensive=expensive)
        reward = result.get()
        return reward
