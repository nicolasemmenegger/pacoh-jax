from typing import Union, Any, Callable

import jax
from jax import numpy as jnp, vmap
from jax.scipy.linalg import cho_solve, cho_factor
from numpyro import distributions as npd
from numpyro.distributions import Categorical, MixtureSameFamily, Independent, Normal, MultivariateNormal

from pacoh.util.typing_util import Tree


def get_mixture(pred_dists, n):
    cat = Categorical(probs=jnp.ones((n,)) / n)
    return MixtureSameFamily(cat, pred_dists)


def get_diagonal_gaussian(loc, scale, event_dims=None) -> Independent:
    if event_dims is None:
        event_dims = loc.ndim
    assert loc.shape == scale.shape and loc.ndim >= event_dims
    return Independent(Normal(loc, scale), event_dims)


def multivariate_kl(dist1: MultivariateNormal, dist2: MultivariateNormal) -> float:
    # TODO this could probably be sped up a bit by combining certain cholesky factorizations
    logdets = jnp.linalg.slogdet(dist2.covariance_matrix)[1] - jnp.linalg.slogdet(dist1.covariance_matrix)[1]
    d2chol = cho_factor(dist2.covariance_matrix)
    trace = jnp.trace(cho_solve(d2chol, dist1.covariance_matrix))
    locdiff = dist2.loc - dist1.loc
    bilin = locdiff @ cho_solve(d2chol, locdiff)

    kl = logdets - dist1.loc.shape[0] + trace + bilin
    return 0.5 * kl


def is_gaussian_dist(dist):
    """Checks whether is a Gaussian distribution we support as public interface."""
    return is_diagonal_gaussian_dist(dist) or isinstance(dist, MultivariateNormal)


def is_diagonal_gaussian_dist(dist):
    """Checks whether is a Gaussian distribution we support as public interface."""
    return isinstance(dist, Independent) and isinstance(dist.base_dist, Normal)


def diagonalize_gaussian(dist):
    if isinstance(dist, Independent) or (
        isinstance(dist, MixtureSameFamily) and isinstance(dist.component_distribution, Independent)
    ):
        return get_diagonal_gaussian(dist.mean, jnp.sqrt(dist.variance))
    elif isinstance(dist, MultivariateNormal):
        return get_diagonal_gaussian(dist.mean, jnp.sqrt(jnp.diag(dist.covariance_matrix)))
    else:
        return get_diagonal_gaussian(dist.mean, jnp.sqrt(dist.variance))
        raise ValueError("unknown argument type" + str(dist))


DIST_ID = [npd.Independent, npd.Normal, npd.MultivariateNormal]  # TODO add others, see how they behave


def _auxiliary_to_jax_type(pytree):
    converted = jax.tree_map(lambda leaf: DIST_ID.index(leaf) if leaf in DIST_ID else leaf, pytree)
    blueprint = jax.tree_map(lambda leaf: leaf in DIST_ID, pytree)
    return converted, blueprint


def _restore_auxiliary(converted, blueprint):
    return jax.tree_multimap(
        lambda leaf, do_restore: DIST_ID[leaf] if do_restore else leaf,
        converted,
        blueprint,
    )


def _flatten_leaf(maybe_dist: Union[npd.Distribution, Any]):
    """If dist is of type npd.Distribution, then return the flattened version, otherwise just return some placeholders"""
    """We also handle the special case where the first argument is a distribution, and the second argument is hk.State"""
    if isinstance(maybe_dist, npd.Distribution):
        params, aux = maybe_dist.tree_flatten()
        return (
            DIST_ID.index(maybe_dist.__class__),
            params,
            _auxiliary_to_jax_type(aux),
        )
    else:
        return (
            -1,
            maybe_dist,
            None,
        )  # id -1 means it's not a distribution, None auxiliary data needed


def _unstack_flat_leaf_pytrees(isleaf, stacked_tree):
    cls_id_tree = jax.tree_multimap(lambda _, stacked_leaf: stacked_leaf[0], isleaf, stacked_tree)
    params = jax.tree_multimap(lambda _, stacked_leaf: stacked_leaf[1], isleaf, stacked_tree)
    translated_aux = jax.tree_multimap(lambda _, stacked_leaf: stacked_leaf[2], isleaf, stacked_tree)
    return cls_id_tree, params, translated_aux


def _unflatten_leaf(cls_id, vmapped_output, aux_converted):
    if cls_id == -1:
        return vmapped_output
    else:
        cls = DIST_ID[cls_id]
        aux = _restore_auxiliary(*aux_converted)
        return cls.tree_unflatten(aux, vmapped_output)


def vmap_dist(f: Callable[[Any], Tree], in_axes=0, out_axes=0, axis_name=None):
    """Helper function that vmaps a function that may return a distribution as output."""

    def flat_f(*args, **kwargs):
        """Returns a 3-tuple of trees:
          * A tree of integers having exactly the leaves of the original output tree, except that numpyro.Distributions are treated as leaves
          * A tree of actual data, where leaves that were numpyro.Distributions were flattened, and therefore correspond to new pytrees
          * A tree of auxiliary information, on how to unflatten the leaves corresponding to flattened Distribution objects
        Note: the tree of actual data has the same prefix definition as the original tree, and hence we can apply vmap with
        custom out_axes!
        """

        output_tree = f(*args, **kwargs)
        flattened_leaves_tree = jax.tree_map(
            _flatten_leaf, output_tree, is_leaf=lambda l: isinstance(l, npd.Distribution)
        )

        # need to know where the leaves are at to unstack the tree correctly
        are_leaves = jax.tree_map(
            lambda _: True, output_tree, is_leaf=lambda l: isinstance(l, npd.Distribution)
        )

        # TODO possibly insert check whether out_axes is valid and does not go into distribution leaves ??
        return _unstack_flat_leaf_pytrees(are_leaves, flattened_leaves_tree)

    vmapped = vmap(
        flat_f,
        in_axes=in_axes,
        out_axes=(None, out_axes, None),
        axis_name=axis_name,
    )

    def unflattened(*args, **kwargs):
        trees = vmapped(*args, **kwargs)
        # the tree of class ids makes sure we stop at the right depth, even though we are not technically at
        # what jax considers a leaf in both the data/param and aux tree
        unflat = jax.tree_multimap(lambda *leaves: _unflatten_leaf(*leaves), *trees)
        return unflat

    return unflattened
