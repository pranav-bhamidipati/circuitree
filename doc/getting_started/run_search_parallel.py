from datetime import datetime
from pathlib import Path
from bistability_parallel import ParallelBistabilityTree


def main(
    n_steps: int = 10_000,
    n_threads: int = 8,
    expensive: bool = True,
    save_dir: str | Path = Path("./tree-backups"),
):
    """Runs a search for bistable circuits using MCTS."""

    # Make a folder for backups
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Search in parallel
    print("Running an MCTS search in parallel (see tutorial notebook #2)...")
    tree = ParallelBistabilityTree(root="ABC::")
    tree.search_mcts_parallel(
        n_steps=n_steps, n_threads=n_threads, run_kwargs={"expensive": expensive}
    )
    print("Search complete!")

    # Save the graph to a GML file and the other attributes to a JSON file
    today = datetime.now().strftime("%y%m%d")
    save_stem = save_dir.joinpath(f"{today}_parallel_bistability_search_step{n_steps}")
    gml_file = save_stem.with_suffix(".gml")
    json_file = save_stem.with_suffix(".json")

    print(f"Saving final tree to {gml_file} and {json_file}")
    tree.to_file(gml_file, json_file)

    print("Done")


if __name__ == "__main__":
    main()
