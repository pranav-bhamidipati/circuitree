from pathlib import Path
from datetime import datetime
from bistability import BistabilityTree


def main(
    n_steps: int = 50_001,
    callback_every: int = 500,
    save_dir: str | Path = Path("./tree-backups"),
    expensive: bool = False,
):
    """Runs a search for bistable circuits using MCTS and saves the tree at regular
    intervals."""

    # Make a folder for backups
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%y%m%d")

    ## Callbacks should have the following call signature:
    ##       callback(tree, iteration, selection_path, simulated_node, reward)
    ## We only need the first two arguments to do a backup.
    def save_tree_callback(tree: BistabilityTree, iteration: int, *args, **kwargs):
        """Saves the BistabilityTree to two files, a `.gml` file containing the
        graph and a `.json` file with the other object attributes."""
        gml_file = save_dir.joinpath(f"{today}_bistability_search_{iteration}.gml")
        json_file = save_dir.joinpath(f"{today}_bistability_search_{iteration}.json")
        tree.to_file(gml_file, json_file)

    # Search with periodic backup
    print("Running an MCTS search for bistable circuits (see tutorial notebook #1)...")
    tree = BistabilityTree(root="ABC::")
    tree.search_mcts(
        n_steps=n_steps,
        progress_bar=True,
        run_kwargs={"expensive": expensive},
        callback=save_tree_callback,
        callback_every=callback_every,
        callback_before_start=False,
    )
    print("Search complete!")


if __name__ == "__main__":
    main()
