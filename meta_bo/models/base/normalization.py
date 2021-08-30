import numpy as np
from typing import Dict


class DataNormalizer:
    """ A class that abstracts away common data normalization tasks. """
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def compute_normalization_stats(self, xs, ys):
        """Computes normalization statistics from the given dataset. """
        pass

    def normalize_data(self, xs, ys):
        """Computes normalization based on the stored normalization statistics. """
        pass

    def set_normalization_stats(self, stats: Dict[str,  np.ndarray]):
        """Set the normalization stats. """
        pass



