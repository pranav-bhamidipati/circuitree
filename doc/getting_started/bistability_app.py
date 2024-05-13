from celery import Celery
from circuitree.models import SimpleNetworkGrammar
import numpy as np
import redis
import os

from bistability import get_bistability_reward

# Specify the address of the Redis database used by Celery
database_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
database = redis.Redis.from_url(database_url)
if database.ping():
    print(f"Connected to Redis database at {database_url}")
else:
    raise ConnectionError(f"Could not connect to Redis database at {database_url}")

# Create a Celery app
app = Celery("bistability", broker=database_url, backend=database_url)

# Define the grammar so it can be accessed by workers
grammar = SimpleNetworkGrammar(["A", "B"], ["activates", "inhibits"])


@app.task
def get_reward_celery(state: str, seed: int, expensive: bool = False) -> float:
    """Returns a reward value for the given state based on how many types of positive
    feedback loops (PFLs) it contains. Same as `BistabilityTree.get_reward()`,
    except this function is evaluated by a Celery worker."""

    # Celery cannot pass Numpy random generators, so we generate a new high-quality
    # seed from the given seed and use it to create a new generator
    hq_seed = np.random.SeedSequence(seed).generate_state(1)[0]
    rg = np.random.default_rng(hq_seed)

    # Compute the reward
    reward = get_bistability_reward(state, grammar, rg, expensive)
    return reward
