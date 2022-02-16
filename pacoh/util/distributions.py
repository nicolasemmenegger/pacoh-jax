from typing import Union, Any, Callable

from jax import numpy as jnp, vmap
from jax.scipy.linalg import cho_solve, cho_factor
from numpyro import distributions as npd
from numpyro.distributions import Categorical, MixtureSameFamily, Independent, Normal, MultivariateNormal


def get_mixture(pred_dists, n):
    # pred_dists = stack_distributions(pred_dists)
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
        raise ValueError("unknown argument type")


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


def _flatten_dist(maybe_dist: Union[npd.Distribution, Any]):
    """If dist is of type npd.Distribution, then return the flattened version, otherwise just return some placeholders"""
    if isinstance(maybe_dist, npd.Distribution):
        params, aux = maybe_dist.tree_flatten()
        return params, *_auxiliary_to_jax_type(aux), DIST_ID.index(maybe_dist.__class__)
    else:
        return maybe_dist, None, None, -1  # id -1 means it's not a distribution


def vmap_dist(f: Callable[[Any], npd.Distribution], in_axes=0, out_axes=0, axis_name=None):
    """Helper function that vmaps a function that may return a distribution as output."""
    flat_f = lambda *args, **kwargs: _flatten_dist(
        f(*args, **kwargs)
    )  # this returns the flat_dist, type tuple
    vmapped = vmap(
        flat_f,
        in_axes=in_axes,
        out_axes=(out_axes, None, None, None),
        axis_name=axis_name,
    )

    def unflattened(*args, **kwargs):
        vmapped_output, aux_converted, aux_blueprint, cls_id = vmapped(*args, **kwargs)
        if cls_id == -1:
            return vmapped_output
        else:
            # function actually returns a distribution
            cls = DIST_ID[cls_id]
            aux = _restore_auxiliary(aux_converted, aux_blueprint)
            return cls.tree_unflatten(aux, vmapped_output)

    return unflattened


if __name__ == "__main__":
    import jax

    def get_dist(mean):
        return Independent(Normal(mean, jnp.ones_like(mean)), 2)

    def get_var(mean):
        return Independent(Normal(mean, jnp.ones_like(mean)), 2).variance

    def get_nondiag_dist(mean):
        return MultivariateNormal(
            mean, jnp.diag(jnp.ones_like(mean)) + 0.001 * jnp.ones((mean.shape[0], mean.shape[0]))
        )

    get_dists = vmap_dist(get_dist)
    get_vars = vmap_dist(get_var)
    get_nondiag_dists = vmap_dist(get_nondiag_dist)
    key = jax.random.PRNGKey(42)
    mean = jax.random.normal(key, (8, 1))
    means = jax.random.normal(key, (3, 8, 1))

    simple_dist = get_dist(mean)  ##  .get_numpyro_distribution()
    batched_dist = get_dists(means)  ## .get_numpyro_distribution()

    simple_var = get_var(mean)  ##  .get_numpyro_distribution()
    batched_vars = get_vars(means)  ## .get_numpyro_distribution()

    multi_dist = get_nondiag_dist(mean.flatten())
    multi_dists = get_nondiag_dists(means.reshape((3, 8)))

    print("vmapped diagonal normal distributions: ")
    print(simple_dist.batch_shape, "&", simple_dist.event_shape)  # prints: () & (8,1)
    print(batched_dist.batch_shape, "&", batched_dist.event_shape)  # Should print: (3,) & (8,1)

    print("can still vmap functions returning something else: ")
    print(simple_var.shape, batched_vars.shape)

    print("can also return multivariate normals in a batched manner")
    print(multi_dist.batch_shape, "&", multi_dist.event_shape)  # prints: () & (8,)
    print(multi_dists.batch_shape, "&", multi_dists.event_shape)  # Should print: (3,) & (8,)
    print(multi_dists.sample(jax.random.PRNGKey(42)))
