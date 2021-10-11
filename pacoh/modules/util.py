from typing import Optional, TypeVar, Collection

import jax
import numpy as np
import os
import logging

import numpyro.distributions
from absl import flags
import warnings
import torch
from jax import numpy as jnp

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


""" ----- Tree Utilities ----- """
Tree = TypeVar('Tree')

# Courtesy of https://github.com/bryanhpchiang/rt/blob/master/utils/transforms.py#L5
def pytrees_stack(pytrees: Collection[Tree], axis=0):
    results = jax.tree_multimap(
        lambda *values: jax.numpy.stack(values, axis=axis), *pytrees)
    return results

def pytree_unstack(pytree: Tree, n=None):
    if n is None:
        n = jax.tree_leaves(pytree)[0].shape[0]

    return [jax.tree_map(lambda x: x[i], pytree) for i in range(n)]

def broadcast_params(tree: Tree, num_repeats):
    return jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), repeats=num_repeats, axis=0), tree)

def pytree_shape(tree: Tree):
    return jax.tree_map(lambda p: p.shape, tree)

def pytree_sum(tree: Tree):
    tree =  tree
    return jax.tree_util.tree_reduce(lambda x, s: x+s, tree, 0.0)

""" ----- Distribution Transformations ------ """
def stack_distributions(distribution: numpyro.distributions.Normal):
    """ Takes a Normal Distribution with batch_shape (n_particles, *dims) and returns one of (*dims, n_particles) """
    loc = distribution.loc
    scale = distribution.scale

    loc = jnp.stack(loc, axis=-1)
    scale = jnp.stack(scale, axis=-1)
    return numpyro.distributions.Normal(scale=scale, loc=loc)