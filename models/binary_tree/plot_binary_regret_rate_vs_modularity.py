import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


def main(
    data_csv: Path = Path("./data/analysis/binary_tree/binary_tree_search.csv"),
    figsize=(2.5, 2),
    fmt: str = "png",
    dpi: int = 300,
    save: bool = False,
    save_dir: Path = Path("./figures/binary_tree"),
):
    df = pd.read_csv(
        Path(data_csv), dtype={"method": "string", "outcome_codes": "string"}
    )

    # Plot mean rate of regret accumulation vs modularity
    dRdt = r"$\Delta\mathrm{Regret}/\Delta$t"
    cols = ["method", "outcome_codes", "seed", "modularity"]
    regret_rate = df.groupby(cols)["dRegret"].mean()
    regret_rate.name = dRdt
    regret_rate = regret_rate.reset_index()
    plot_data = regret_rate.loc[regret_rate["method"] == "mcts"].copy()
    plot_data[r"$\mathcal{M}_T$"] = plot_data["modularity"]

    # Plot regret vs time for a selection of modularity values
    fig = plt.figure(figsize=figsize)
    sns.scatterplot(
        data=plot_data,
        x=r"$\mathcal{M}_T$",
        y=dRdt,
        s=5,
        palette=["black"],
    )

    # plt.ylabel(r"$\frac{\Delta \mathrm{Regret}}{\Delta t}$")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if save:
        save_dir = Path(save_dir)
        fname = save_dir.joinpath(f"regret_rate_vs_modularity").with_suffix(f".{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(str(fname), dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    main(
        # save=True,
    )
