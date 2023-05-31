from pathlib import Path
from typing import Iterable
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd

from models.oscillation.oscillation import OscillationTree


def main(
    save: bool = False,
    nodes_fname: str = "3tf_dag_nodes_xy.csv",
    edges_fname: str = "3tf_dag_edges_xy.csv",
    data_dir: str = "data",
    components: Iterable[str] = ["A", "B", "C"],
    interactions: Iterable[str] = ["activates", "inhibits"],
    root: str = "ABC::",
):
    dag = OscillationTree(
        components=components, interactions=interactions, root=root, tree_shape="dag"
    )
    dag.grow_tree(n_visits=1)
    D = dag.complexity_graph_from_bfs(dag.graph)[0]

    pos = graphviz_layout(D, prog="dot")
    pos_df = pd.DataFrame.from_dict(pos, orient="index", columns=("x", "y"))
    edges_df = pd.DataFrame(D.edges(), columns=["source", "target"])

    d = Path(data_dir)

    if save:
        fpath = d.joinpath(nodes_fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        pos_df.to_csv(fpath)

        fpath = d.joinpath(edges_fname)
        print(f"Writing to: {fpath.resolve().absolute()}")
        edges_df.to_csv(fpath)


if __name__ == "__main__":
    main(
        save=True,
    )
