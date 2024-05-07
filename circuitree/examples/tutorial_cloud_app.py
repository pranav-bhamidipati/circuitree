from celery import Celery
from models import SimpleNetworkGrammar
import numpy as np
import os
from time import sleep

from circuitree.examples.sequential import get_bistability_reward

# Use a Redis server hosted on the cloud
database_url = os.environ["CIRCUITREE_CLOUD_REDIS_URL"]
if not database_url:
    raise ValueError(
        "Please set the CIRCUITREE_CLOUD_REDIS_URL environment variable "
        "to the URL of a Redis server."
    )
app = Celery("bistability", broker=database_url, backend=database_url)

# Define the grammar here also so it can be accessed by workers
grammar = SimpleNetworkGrammar(["A", "B", "C"], ["activates", "inhibits"])


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
