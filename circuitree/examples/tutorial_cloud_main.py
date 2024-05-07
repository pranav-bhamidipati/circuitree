from celery import Celery
import os

# Use a Redis server hosted on the cloud
database_url = os.environ["CIRCUITREE_CLOUD_REDIS_URL"]
if not database_url:
    raise ValueError(
        "Please set the CIRCUITREE_CLOUD_REDIS_URL environment variable "
        "to the URL of a Redis server."
    )
app = Celery("bistability_main", broker=database_url, backend=database_url)


@app.task
def run_mcts_parallel(
    n_steps: int, n_threads: int, expensive: bool = True, root="ABC::", **kwargs
) -> tuple[dict, str]:
    """Runs a parallel MCTS search with the given number of steps and threads. Returns
    a dictionary of attributes (from `CircuiTree.to_dict()`) and the search graph object
    as a string in GML format. Together they can be used to reconstruct the search tree
    using `CircuiTree.from_file()`."""

    from examples.distributed import DistributedBistabilityTree

    # Run the search in parallel
    tree = DistributedBistabilityTree(root=root, **kwargs)
    tree.search_mcts_parallel(
        n_steps=n_steps,
        n_threads=n_threads,
        run_kwargs=dict(expensive=expensive),
        **kwargs
    )
    return tree.to_string()
