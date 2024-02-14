from pathlib import Path

import yaml


def read_yaml(fname: str | Path) -> dict:
    """Reads a YAML file and returns the data as dictionary."""
    fname = Path(fname)
    with fname.open("r", encoding="utf8") as file:
        return yaml.safe_load(file)
