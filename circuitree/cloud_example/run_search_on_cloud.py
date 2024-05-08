from typing import Optional
from time import perf_counter


def main(
    n_steps: int,
    n_threads: int,
    expensive: bool = True,
    root="ABC::",
    run_kwargs: Optional[dict] = None,
    **kwargs,
) -> tuple[dict, str]:

    from cloud_app import app, worker_tasks, main_tasks
    from bistability import DistributedBistabilityTree

    # Time the search
    print("Running parallel search...")
    start_time = perf_counter()

    print(main_tasks)
    print(main_tasks.run_mcts_parallel)

    ...
    
    result = main_tasks.run_mcts_parallel.delay(
        n_steps=n_steps,
        n_threads=n_threads,
        expensive=expensive,
        root=root,
        run_kwargs=run_kwargs,
        **kwargs,
    )
    serialized = result.get()

    end_time = perf_counter()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Save the search tree to a file
    tree = DistributedBistabilityTree.from_file(*serialized)
    tree.to_file("search_tree.gml", "attrs.json")


if __name__ == "__main__":
    main(
        n_steps=24,
        n_threads=8,
    )
