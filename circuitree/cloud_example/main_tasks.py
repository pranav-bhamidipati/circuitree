from celery import shared_task
from typing import Optional
from bistability import DistributedBistabilityTree

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
