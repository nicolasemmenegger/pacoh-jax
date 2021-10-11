import warnings

import numpy as np
from typing import Dict, NamedTuple
import jax
from jax import numpy as jnp

from pacoh.modules.util import _handle_batch_input_dimensionality


class Statistics(NamedTuple):
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

class DataNormalizer:
    """ A class that abstracts away common data storage and normalization tasks """
    def __init__(self, input_dim, normalization_stats: Statistics = None):
        """ The standard constructor
        """
        self.input_dim = input_dim
        if normalization_stats is not None:
            self.stats = normalization_stats
        else:
            self.stats = DataNormalizer.get_trivial_stats()

    def get_trivial_stats(self):
        return Statistics(
                x_mean=np.ones((self.input_dim,)),
                x_std=np.ones((self.input_dim,)),
                y_mean=np.zeros((1,)),
                y_std=np.ones((1,)))

    def from_meta_data_sets(cls, input_dim, meta_tasks):
        """Second constructor that directly infers the normalisation stats from the meta_data_set"""
        stats = cls.compute_nomalization_stats_from_meta_datasets(meta_tasks)
        return cls(input_dim, stats)

    @classmethod
    def compute_normalization_stats(cls, xs, ys):
        """Computes normalization statistics from the given dataset. """
        pass

    @classmethod
    def compute_normalization_stats_meta(cls, meta_train_tuples):
        """
        Expects y to be flattened
        """
        xs_stack, ys_stack = map(list,
                                 zip(*[_handle_batch_input_dimensionality(x_train, y_train) for x_train, y_train in
                                       meta_train_tuples]))
        all_xs, all_ys = np.concatenate(xs_stack, axis=0), np.concatenate(ys_stack, axis=0)

        if self.normalize_data:
            return Statistics(
                x_mean=np.mean(all_xs, axis=0),

                y_mean=np.mean(all_ys, axis=0)
            )
            self.x_mean, self.y_mean = np.mean(all_xs, axis=0), np.mean(all_ys, axis=0)
            self.x_std, self.y_std = np.std(all_xs, axis=0) + 1e-8, np.std(all_ys, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(all_xs.shape[1]), np.zeros(1)
            self.x_std, self.y_std = np.ones(all_xs.shape[1]), np.ones(1)

    def normalize_data(self, xs, ys):
        """Computes normalization based on the stored normalization statistics. """

    def set_normalization_stats(self, stats: Dict[str,  np.ndarray]):
        """Set the normalization stats. """
        pass



class Sampler:
    def __init__(self, xs, ys, batch_size, rds):
        self.num_batches = xs.shape[0] // batch_size
        self._rds, key = jax.random.split(rds)

        ids = jnp.arange(xs.shape[0])
        perm = jax.random.permutation(key, ids)
        self.i = -1
        self.xs = xs[perm]
        self.ys = ys[perm]
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        self.i = (self.i + 1) % self.num_batches
        start = self.i * self.batch_size
        end = start + self.batch_size
        return self.xs[start:end], self.ys[start:end]





