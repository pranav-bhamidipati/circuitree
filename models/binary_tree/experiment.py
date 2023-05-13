from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

from binary_tree import binary_tree_search


config_dir = (
    "./models/binary_tree"  # Location of config file for experiment (e.g. config.json)
)
storage_dir = "./data/binary_tree/runs"  # Store data locally
# storage_dir = "/home/pbhamidi/scratch/circuitree/sacred"  # Store on HPC scratch

ex = Experiment("binary_tree_search")
ex.add_config(str(Path(config_dir).joinpath("config.json")))

sacred_storage_dir = Path(storage_dir)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(sacred_storage_dir))


@ex.automain
def run_one(_config, _run, seed):
    from sys import argv

    ex.add_source_file(argv[0])
    binary_tree_search(ex=ex, **_config)
