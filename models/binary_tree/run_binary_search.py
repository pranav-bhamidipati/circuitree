from functools import partial
from itertools import chain, product
from multiprocessing import Pool
from more_itertools import batched, chunked_even
from psutil import cpu_count
from typing import Iterable, Mapping, Optional
import numpy as np

from binary_tree import idx_to_outcome
from circuitree import vround


def make_inputs(
    depth: int,
    methods: tuple[str],
    n_swaps_from_sorted: Mapping[int, int],
    n_swaps_from_mixed: Mapping[int, int],
    n_successes: int,
    n_replicates: int,
    n_shuffles: int = 0,
    seed0: int = 0,
    maxiter: int = 10000,
):
    """Generate permuted leaves of a binary tree with a given depth and number of successful outcomes."""
    rg = np.random.default_rng(seed0)
    n_outcomes = 2**depth
    n_failures = n_outcomes - n_successes

    # Get permutations of successes and failures starting from the most sorted tree
    failures, successes = np.split(np.arange(n_outcomes), [n_failures])
    reached_maxiter = True
    outcomes = set()
    for n_swap, n_samples in n_swaps_from_sorted.items():
        k = 0
        for _ in range(maxiter):
            # Randomly swap 1s and 0s
            remove = rg.choice(n_successes, size=n_swap, replace=False)
            add = rg.choice(n_failures, size=n_swap, replace=False)
            succ = successes.copy()
            succ[remove] = failures[add]
            out = idx_to_outcome(succ, depth)
            if out not in outcomes:
                k += 1
                outcomes.add(out)
            if k >= n_samples:
                reached_maxiter = False
                break
        if reached_maxiter:
            print(f"Reached maxiter: {maxiter}")

    # Get permutations of successes and failures starting from the most mixed tree
    mixed_failures = vround(np.linspace(0, n_outcomes - 1, n_failures))
    mixed_successes = np.array(
        [i for i in range(n_outcomes) if i not in mixed_failures]
    )

    for n_swap, n_samples in n_swaps_from_mixed.items():
        k = 0
        for _ in range(maxiter):
            # Randomly swap 1s and 0s
            remove = rg.choice(n_successes, size=n_swap, replace=False)
            add = rg.choice(n_failures, size=n_swap, replace=False)
            msucc = mixed_successes.copy()
            msucc[remove] = mixed_failures[add]
            out = idx_to_outcome(msucc, depth)
            if out not in outcomes:
                k += 1
                outcomes.add(out)
            if k >= n_samples:
                reached_maxiter = False
                break
        if reached_maxiter:
            print(f"Reached maxiter: {maxiter}")

    outcome = next(iter(outcomes))
    for _ in range(n_shuffles):
        outcome = "".join(rg.permutation(list(outcome)))
        outcomes.add(out)

    tree_outcomes = sorted(outcomes)

    return product(methods, tree_outcomes, range(seed0, seed0 + n_replicates))


def run_one(
    args: tuple,
    N: int,
    save: bool,
    estimate_modularity: bool,
    **kwargs,
):
    method, outcome_code, seed = args

    from experiment import ex

    cfg_updates = {
        "method": method,
        "outcome_code": outcome_code,
        "seed": seed,
        "N": N,
        "save": save,
        "estimate_modularity": estimate_modularity,
    }
    ex.run(config_updates=cfg_updates)


def run_batch(inputs, **kwargs):
    return [run_one(args, **kwargs) for args in inputs]


def main(
    depth: int = 5,
    estimate_modularity: bool = False,
    methods: tuple[str] = ("mcts", "sequential"),
    n_swaps_from_sorted: Mapping[int, int] = {2: 16, 3: 36},
    n_swaps_from_mixed: Mapping[int, int] = {0: 1, 1: 16},
    n_shuffles: int = 15,
    n_successes: int = 16,
    n_replicates: int = 10,
    N: int = 10000,
    nthreads: Optional[int] = None,
    logical: bool = False,
    batch: bool = True,
    save: bool = False,
    seed0: int = 0,
    maxiter: int = 10000,
    **kwargs,
):
    nthreads = nthreads or cpu_count(logical=logical)

    inputs = list(
        make_inputs(
            depth,
            methods,
            n_swaps_from_sorted,
            n_swaps_from_mixed,
            n_successes,
            n_replicates,
            n_shuffles=n_shuffles,
            seed0=seed0,
            maxiter=maxiter,
        )
    )
    chunksize, mod = divmod(len(inputs), nthreads)
    chunksize += 1 if mod else 0
    batched_inputs = list(chunked_even(inputs, chunksize))

    kwargs.update(dict(N=N, save=save, estimate_modularity=estimate_modularity))

    if nthreads == 1:
        if batch:
            for b in batched_inputs:
                # results = list(chain(run_batch_with_args(*batch, N=N, save=save)))
                results = list(chain(run_batch(b, **kwargs)))
        else:
            results = [run_one(input, **kwargs) for input in inputs]
    else:
        results = []
        print(f"Assembling pool of {nthreads} workers...")

        if batch:
            run_batch_with_kwargs = partial(run_batch, **kwargs)

            print(f"Running {len(inputs)} inputs in {len(batched_inputs)} batches...")
            with Pool(nthreads) as p:
                for r in p.imap_unordered(run_batch_with_kwargs, batched_inputs):
                    results.append(r)

        else:
            run_one_with_kwargs = partial(run_one, **kwargs)

            print(f"Running {len(inputs)} inputs...")
            with Pool(nthreads) as p:
                for r in p.imap_unordered(run_one_with_kwargs, inputs):
                    results.append(r)

    ...


if __name__ == "__main__":
    # Run a few replicates of sequential search
    # (This is for completeness - they all have the same aggregate statistics)
    main(
        depth=5,
        estimate_modularity=True,
        n_successes=16,
        methods=("sequential",),
        n_swaps_from_sorted={2: 3, 3: 3},
        n_swaps_from_mixed={0: 1, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3},
        n_shuffles=15,
        n_replicates=5,
        N=10000,
        save=True,
        logical=True,
        seed0=2023,
        # batch=False,
        # nthreads=1,
    )

    # Run many replicates of MCTS
    main(
        depth=5,
        estimate_modularity=True,
        n_successes=16,
        methods=("mcts",),
        n_swaps_from_sorted={2: 16, 3: 20},
        n_swaps_from_mixed={0: 1, 1: 4, 2: 10, 3: 10, 4: 20, 5: 30, 6: 30},
        n_shuffles=15,
        n_replicates=5,
        N=10000,
        save=True,
        logical=True,
        seed0=2023,
        # batch=False,
        # nthreads=1,
    )

    ### Args for testing
    # main(
    #     depth=5,
    #     methods=("mcts",),
    #     n_swaps={3: 1},
    #     n_successes=16,
    #     n_replicates=20,
    #     N=50,
    #     nthreads=1,
    #     # save=True,
    #     logical=True,
    # )
