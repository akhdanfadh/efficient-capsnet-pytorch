from itertools import repeat
from pathlib import Path
from typing import Iterator, Optional, Union

import pandas as pd
import yaml

from .data_loader import BaseDataLoader
from .logger import TensorboardWriter


def read_yaml(fname: Union[str, Path]) -> dict:
    """Reads a YAML file and returns the data as dictionary."""
    fname = Path(fname)
    with fname.open("r") as file:
        return yaml.safe_load(file)


def inf_loop(data_loader: BaseDataLoader) -> Iterator[BaseDataLoader]:
    """Wraps a data loader to yield data endlessly.

    This function takes a data loader and returns an infinite iterator over
    the data loader's data. This can be useful when you want to iterate over
    a dataset for an indefinite number of times (non-epoch or iteration-based
    training).

    Args:
        data_loader (BaseDataLoader): The data loader to wrap. This can be an instance
        of BaseDataLoader or any of its subclasses.

    Yields:
        BaseDataLoader: The next data loader in the infinite sequence.
    """
    for loader in repeat(data_loader):
        yield loader


class MetricTracker:
    def __init__(self, *keys: str, writer: Optional[TensorboardWriter] = None):
        """A class used to track metrics for a given set of keys.

        Args:
            *keys (str): Metrics keys to be tracked. Used as index in a DataFrame with columns "total", "counts", "average".
            writer (TensorboardWriter, optional): Used to log the metrics. Defaults to None.
        """
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """Reset all values in the DataFrame to 0."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float, n: int = 1):
        """Updates the total and counts for a given key, then calculates the average.

        Args:
            key (str): The key for which the metrics are updated.
            value (float): The value to add to the total for the key.
            n (int, optional): The count to add to the counts for the key. Defaults to 1.
        """
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data["total"][key] / self._data["counts"][key]
        )

        if self.writer:
            self.writer.add_scalar(key, value)

    def avg(self, key: str) -> float:
        """Returns the average for a given key."""
        return self._data["average"][key]

    def result(self) -> dict:
        """Returns a dictionary of averages for all keys."""
        return dict(self._data["average"])
