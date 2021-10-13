import numpy as np
from typing import Dict, NamedTuple, Optional
import jax
from jax import numpy as jnp


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


""" ----- Dimension Handling ----- """
def _handle_point_input_dimensionality(self, x, y):
    # TODO merge with the _util function and just use that

    if x.ndim == 1:
        assert x.shape[-1] == self.input_dim
        x = x.reshape((-1, self.input_dim))

    if isinstance(y, float) or y.ndim == 0:
        y = np.array(y)
        y = y.reshape((1,))
    elif y.ndim == 1:
        pass
    else:
        raise AssertionError('y must not have more than 1 dim')
    return x, y

def _handle_batch_input_dimensionality(xs: np.ndarray, ys: Optional[np.ndarray] = None, flatten_ys: bool = True):
    """
    Takes a dataset S=(xs,ys) and returns it in a uniform fashion. x shall have shape (num_points, input_dim) and
    y shall have size (num_points), that is, we only consider scalar regression targets.
    Args:
        xs: The inputs
        ys: The labels (optional)
        flatten: Whether to return ys as (num_points), or (num_points, 1)
    Notes:
        ys can be None, to easily handle test data.
    """
    if xs.ndim == 1:
        xs = np.expand_dims(xs, -1)

    assert xs.ndim == 2

    if ys is not None:
        if flatten_ys:
            ys = ys.flatten()
            assert xs.shape[0] == ys.size
            return xs, ys
        else:
            if ys.ndim == 1:
                ys = np.expand_dims(ys, -1)
            assert xs.shape[0] == ys.shape[0], "Number of points and labels is not the same"
            assert ys.ndim == 2


        if flatten:
            return xs, ys.flatten()
        else:
            return xs, ys
    else:
        return xs








