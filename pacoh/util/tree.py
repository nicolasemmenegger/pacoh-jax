from typing import Collection

import jax
from jax import numpy as jnp
import numpyro

from pacoh.util.typing import Tree


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