from typing import Iterable, Optional
import json
from pathlib import Path
from oscillation import OscillationTree


def main(
    save_dir: Path,
    n_components: int = 3,
    interactions: Iterable[str] = ["activation", "inhibition"],
    root: str = "ABC::",
    save: bool = False,
):
    ord_A = ord("A")
    components = [chr(ord_A + i) for i in range(n_components)]
    tree = OscillationTree(components=components, interactions=interactions, root=root)
    tree.grow_tree()
    terminal_genotypes = [n for n in tree.bfs_iterator() if tree.is_terminal(n)]
    ...

    if save:
        fname = Path(save_dir) / f"genotypes_queue.json"
        fname = fname.resolve().absolute()
        print(f"Writing genotype queue to {fname}")
        with fname.open("w") as f:
            json.dump(terminal_genotypes, f)


if __name__ == "__main__":
    save_dir = Path("data/oscillation")
    save_dir.mkdir(exist_ok=True)
    main(
        save_dir,
        save=True,
    )
