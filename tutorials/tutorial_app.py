from celery import Celery
from circuitree.models import SimpleNetworkGrammar
import numpy as np
from time import sleep
from tutorial_1_basic_example import get_bistability_reward

# Specify the address of the Redis server and create a Celery app
broker_url = "redis://localhost:6379"
app = Celery("bistability", broker=broker_url, backend=broker_url)

# Define the grammar here also so it can be accessed by workers
grammar = SimpleNetworkGrammar(["A", "B"], ["activates", "inhibits"])


@app.task
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
