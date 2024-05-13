#######################################################################################
### This must be run before you declare any multithreaded code using gevent, and it must
### be the first lines of the module! This overwrites ("monkey-patches") the Python
### standard library to be compatible with gevent's cooperative threading model.
from gevent import monkey

monkey.patch_all()
#######################################################################################

from bistability import BistabilityTree
from bistability_app import get_reward_celery


class ParallelBistabilityTree(BistabilityTree):
    """This class is identical to BistabilityTree except that it uses Celery to compute
    rewards in parallel. This allows other threads to continue performing MCTS steps
    while one thread waits for its reward calculation to finish."""

    def get_reward(self, state, expensive=True):
        # Generate a random seed and run the task in a Celery worker
        seed = int(self.rg.integers(0, 2**32))
        result = get_reward_celery.delay(state, seed, expensive=expensive)
        reward = result.get()
        return reward
