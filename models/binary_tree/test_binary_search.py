import pandas as pd
from pathlib import Path

# oc = "00000000000000110101111111111111"
# oc = "00000001000000010111011111111111"

data_csv = Path("./data/analysis/binary_tree/binary_tree_search.csv")


def run_ex(oc, seed):
    from experiment import ex

    cfg_updates = {
        "method": "mcts",
        "outcome_code": oc,
        "seed": seed,
        "N": 10000,
        "save": False,
        "estimate_modularity": True,
    }
    ex.run(config_updates=cfg_updates)


if __name__ == "__main__":
    df = pd.read_csv(
        Path(data_csv), dtype={"method": "string", "outcome_codes": "string"}
    )

    df["min_estimate"] = df.groupby(["outcome_codes", "seed"])[
        "modularity_estimates"
    ].transform(lambda x: x == x.min())
    
    # Find where modularity estimate is negative (BAD)
    bugs = df.loc[
        df["modularity_estimates"] < 0 & df["min_estimate"],
        ["outcome_codes", "seed", "t"],
    ].drop_duplicates()

    ...

    oc = bugs.iloc[bugs.t.argmin()]["outcome_codes"]
    seed = int(bugs.iloc[bugs.t.argmin()]["seed"])

    run_ex(oc, seed)

    ...

    for oc, seed in df.values:
        run_ex(oc, seed)
