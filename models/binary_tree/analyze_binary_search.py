import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance


def main(
    load_dir: Path = Path("./data/binary_tree/runs"),
    save: bool = False,
    save_dir: Path = Path("./data/analysis/binary_tree"),
):
    run_dirs = sorted(Path(load_dir).glob("[0-9]*"))

    dfs = []
    for i, run_dir in enumerate(run_dirs):
        config_file = run_dir.joinpath("config.json")
        results_file = run_dir.joinpath("results.hdf5")

        data = {}

        with config_file.open("r") as c:
            config = json.load(c)
            data["method"] = config["method"]

        with h5py.File(str(results_file), "r") as f:
            for k, v in f.items():
                data[k] = np.asarray(v)
            data["N"] = int(data["N"])
            data["depth"] = int(data["depth"])
            data["seed"] = int(data["seed"])
            data["modularity"] = float(data["modularity"])
            data["rewards"] = data["rewards"].astype(int)
            data["modularity_estimates"] = data["modularity_estimates"].astype(float)
            data["outcome_codes"] = str(data["outcome_codes"])[2:-1]

            data["t"] = np.arange(1, data["N"] + 1)

        dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    df["outcome_codes"] = df["outcome_codes"].astype("string")
    df = df.sort_values(["outcome_codes", "seed", "method", "t"])
    df["dRegret"] = 1 - df["rewards"]
    df["regret"] = df.groupby(["outcome_codes", "seed", "method"])["dRegret"].cumsum()

    df["Tree Modularity"] = df["modularity"].apply(lambda i: f"{i:.3f}").values

    least_swaps_outcome = df["outcome_codes"].min()
    n_outcomes = len(least_swaps_outcome)
    n_successes = least_swaps_outcome.count("1")
    sorted_outcome = "0" * (n_outcomes - n_successes) + "1" * n_successes

    # ns1 = hamming_distance_one_to_many(sorted_outcome, df["outcome_codes"].values)
    get_n_swaps = lambda oc: round(
        distance.hamming(tuple(sorted_outcome), tuple(oc)) * len(oc)
    )
    unique_ocs = df["outcome_codes"].unique()
    hamming = pd.DataFrame(
        dict(outcome_codes=unique_ocs, hamming=[get_n_swaps(oc) for oc in unique_ocs])
    )
    df = pd.merge(df, hamming, on="outcome_codes")

    if save:
        csv = Path(save_dir).joinpath("binary_tree_search.csv")
        print(f"Writing to: {csv.resolve().absolute()}")
        df.to_csv(csv, index=False)


if __name__ == "__main__":
    main(
        save=True,
    )
