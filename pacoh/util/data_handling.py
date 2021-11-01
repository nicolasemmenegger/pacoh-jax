import warnings

import numpy as np
from typing import Optional
import jax
from jax import numpy as jnp


class DataNormalizer:
    def __init__(self, input_dim, output_dim, flatten_ys=False, normalize_data=True):
        self.x_mean = jnp.zeros((input_dim,))
        self.x_std = jnp.ones((input_dim,))
        self.y_mean = jnp.zeros((output_dim,))
        self.y_std = jnp.zeros((input_dim,))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.flatten_ys = flatten_ys
        self.turn_off_normalization = not normalize_data

    @classmethod
    def from_meta_tuples(cls, meta_train_tuples, flatten_ys=False, normalize_data=True):
        xs_stack, ys_stack = map(list,
                                 zip(*[handle_batch_input_dimensionality(x_train, y_train) for x_train, y_train in
                                       meta_train_tuples]))
        all_xs, all_ys = np.concatenate(xs_stack, axis=0), np.concatenate(ys_stack, axis=0)

        return cls.from_dataset(all_xs, all_ys, flatten_ys, normalize_data)

    @classmethod
    def from_dataset(cls, xs, ys, flatten_ys=False, normalize_data=True):
        """
        Computes mean and std of the dataset and sets the statistics
        """
        input_dim = xs.shape[-1]
        output_dim = 1 if ys.ndim == 1 else ys.shape[-1]

        normalizer = cls(input_dim, output_dim, flatten_ys, normalize_data)
        normalizer.x_mean, normalizer.y_mean = jnp.mean(xs, axis=0), jnp.mean(ys, axis=0)
        normalizer.x_std, normalizer.y_std = jnp.std(xs, axis=0) + 1e-8, jnp.std(ys, axis=0) + 1e-8

        return normalizer

    def normalize_data(self, xs, ys=None):
        """
        Normalizes the data according to the stored statistics and returns the normalized data
        """
        if self.turn_off_normalization:
            if ys is None:
                return xs
            else:
                return xs, ys

        xs_normalized = (xs - self.x_mean[None, :]) / self.x_std[None, :]
        
        if ys is None:
            return xs_normalized
        else:
            ys_normalized = (ys - self.y_mean) / self.y_std
            return xs_normalized, ys_normalized

    def handle_data(self, xs, ys=None):
        if ys is not None:
            xs, ys = handle_batch_input_dimensionality(xs, ys)
            return self.normalize_data(xs, ys)
        else:
            xs = handle_batch_input_dimensionality(xs)
            return self.normalize_data(xs)


class Sampler:
    def __init__(self, xs, ys, batch_size, rds, shuffle=True):
        self.num_batches = xs.shape[0] // batch_size
        self.shuffle = shuffle
        self._rng = rds
        if shuffle:
            self._rng, shuffle_key = jax.random.split(self._rng)
            ids = jnp.arange(xs.shape[0])
            perm = jax.random.permutation(shuffle_key, ids)
            self.i = -1
            self.xs = xs[perm]
            self.ys = ys[perm]
        else:
            self.xs = xs
            self.ys = ys

        self.batch_size = batch_size
        warnings.warn(
            "this currently does not support a scenario in which the dataset size is not divisible by the batch size")

    def __iter__(self):
        return self

    def __next__(self):
        if self.shuffle:
            # just iterate
            self.i = (self.i + 1) % self.num_batches
            start = self.i * self.batch_size
            end = start + self.batch_size
            return self.xs[start:end], self.ys[start:end]
        else:
            # just subsample at random using the _rds
            raise NotImplementedError("only shuffle mode is supported for now")


def handle_point_input_dimensionality(self, x, y):
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


def handle_batch_input_dimensionality(xs: np.ndarray, ys: Optional[np.ndarray] = None, flatten_ys: bool = True):
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
            return xs, ys
    else:
        return xs