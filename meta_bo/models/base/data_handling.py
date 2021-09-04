import numpy as np
from typing import Dict, NamedTuple

class Statistics(NamedTuple):
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

class DataNormalizer:
    """ A class that abstracts away common data storage and normalization tasks """
    def __init__(self, input_dim, normalization_stats: Statistics = None):
        """ The standard constructur
        """
        self.input_dim = input_dim
        if normalization_stats is not None:
            self.stats = normalization_stats
        else:
            self.stats = Statistics(
                x_mean=np.ones((self.input_dim,)),
                x_std=np.ones((self.input_dim,)),
                y_mean=1.0,
                y_std=1.0
            )

    @classmethod
    def from_meta_data_sets(cls, input_dim, meta_tasks):
        """Second constructor that directly infers the normalisation stats from the meta_data_set"""
        stats = cls.compute_nomalization_stats_from_meta_datasets(meta_tasks)
        return cls(input_dim, stats)

    @classmethod
    def compute_normalization_stats(cls, xs, ys):
        """Computes normalization statistics from the given dataset. """
        pass

    @classmethod
    def compute_normalization_stats_from_meta_datasets(cls, meta_tasks):
        pass

    def normalize_data(self, xs, ys):
        """Computes normalization based on the stored normalization statistics. """

    def set_normalization_stats(self, stats: Dict[str,  np.ndarray]):
        """Set the normalization stats. """
        pass







