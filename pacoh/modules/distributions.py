from abc import ABC, abstractmethod
from collections import Callable
from typing import Any

import numpyro
from numpyro.distributions import Categorical, MixtureSameFamily, Independent, MultivariateNormal, Normal
import numpyro.distributions as npd
from jax import numpy as jnp, tree_util
import haiku as hk
from jax.scipy.linalg import cho_solve, cho_factor

from pacoh.modules.common import PositiveParameter
from pacoh.util.constants import LIKELIHOOD_MODULE_NAME


class VmappableDistribution(ABC):
    @abstractmethod
    def get_numpyro_distribution(self):
        pass

    @abstractmethod
    def tree_flatten(self):
        pass
        # base_flatten, base_aux = self.base_dist.tree_flatten()
        # return base_flatten, (
        #     type(self.base_dist),
        #     base_aux,
        #     self.reinterpreted_batch_ndims,
        # )

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data, params):
        pass

    # register Distribution as a pytree
    # ref: https://github.com/google/jax/issues/2916
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tree_util.register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

        # base_cls, base_aux, reinterpreted_batch_ndims = aux_data
        # base_dist = base_cls.tree_unflatten(base_aux, params)
        # return cls(base_dist, reinterpreted_batch_ndims)


# The following distributions are a small hack that allows us to vmap functions returning predictive distributions
class DiagonalGaussian(VmappableDistribution):
    def __init__(self, loc, scale, event_dims=2):
        self.loc = loc
        self.scale = scale
        self.event_dims = event_dims

    def get_numpyro_distribution(self):
        return npd.Independent(npd.Normal(self.loc, self.scale), self.event_dims)

    def tree_flatten(self):
        return (self.loc, self.scale), self.event_dims  # flattened_dist, aux
        # Also, dear jax, why are the return values reverse of args of tree_unflatten??

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        # aux_data is event_dims
        loc, scale = params
        return cls(loc, scale, aux_data)


# class DiagonalGaussianMixture(DiagonalGaussian):
#     def get_numpyro_distribution(self):
#         return get_mixture(super().get_numpyro_distribution(),


# class MultivariateGaussian:
#     def __init__(self, loc, covariance):
#         self.loc = loc
#         self.covariance = covariance
#
#     def get_numpyro_distribution(self):
#         return npd.MultivariateNormal(self.loc, self.scale), self.event_dims)
#
#     def tree_flatten(self):
#         return (self.loc, self.covariance), None
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, params):
#         # aux_data is event_dims
#         return cls(*params, aux_data)


class JAXGaussianLikelihood(hk.Module):
    def __init__(
        self,
        variance: float = 1.0,
        variance_constraint_gt=0.0,
        output_dim=1,
        learn_likelihood=True,
    ):
        super().__init__(LIKELIHOOD_MODULE_NAME)
        variance = jnp.ones((output_dim,)) * variance
        if learn_likelihood:
            self.variance = PositiveParameter(variance, boundary_value=variance_constraint_gt)
        else:
            self.variance = lambda: variance

    def __call__(self, posterior):
        if isinstance(posterior, jnp.ndarray):
            # just an array of means
            new_scale = jnp.sqrt(self.variance()) * jnp.ones_like(posterior)
            return get_diagonal_gaussian_vmappable(posterior, new_scale)
            # return numpyro.distributions.Normal(loc=posterior, scale=new_scale)
        elif isinstance(posterior, numpyro.distributions.MultivariateNormal):
            raise NotImplementedError
            d = posterior.loc.shape[0]
            cov_with_noise = posterior.covariance_matrix + jnp.eye(d, d) * self.variance()
            return numpyro.distributions.MultivariateNormal(
                loc=posterior.loc, covariance_matrix=cov_with_noise
            )
        elif isinstance(posterior, DiagonalGaussian):
            scale = jnp.sqrt(jnp.square(posterior.scale) + self.variance())
            return get_diagonal_gaussian_vmappable(posterior.base_dist.loc, scale)
        else:
            raise ValueError(
                "posterior should be either a multivariate diagonal, full covariance gaussian or an "
                + "array of means"
            )

    def log_prob(self, ys_true, ys_pred):
        gauss = get_diagonal_gaussian_vmappable(
            jnp.zeros_like(ys_true), jnp.sqrt(self.variance()) * jnp.ones_like(ys_true)
        ).get_numpyro_distribution()
        return gauss.log_prob(ys_pred - ys_true)


def get_mixture(pred_dists, n):
    # pred_dists = stack_distributions(pred_dists)
    cat = Categorical(probs=jnp.ones((n,)) / n)
    return MixtureSameFamily(cat, pred_dists)


def get_diagonal_gaussian_vmappable(loc, scale) -> DiagonalGaussian:
    # Due to https://github.com/pyro-ppl/numpyro/issues/1317, we use our own vmappable distributions here
    assert loc.shape == scale.shape and loc.ndim <= 2
    return DiagonalGaussian(loc, scale, loc.ndim)  # the resulting distribution has trivial batch_shape


def get_diagonal_gaussian_numpyro(loc, scale, event_dims=None) -> Independent:
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


def diagonalize_gaussian(dist):
    if isinstance(dist, Independent) or (
        isinstance(dist, MixtureSameFamily) and isinstance(dist.component_distribution, Independent)
    ):
        return dist
    elif isinstance(dist, MultivariateNormal):
        return get_diagonal_gaussian_numpyro(dist.mean, jnp.sqrt(jnp.diag(dist.covariance_matrix)))
    else:
        raise ValueError("unknown argument type")


DIST_ID = [npd.Independent, npd.Normal]  # TODO add others, see how they behave

def _auxiliary_to_jax_type(pytree):
    converted = jax.tree_map(lambda leaf: DIST_ID.index(leaf) if leaf in DIST_ID else leaf, pytree)
    blueprint = jax.tree_map(lambda leaf: leaf in DIST_ID, pytree)
    return converted, blueprint

def _restore_auxiliary(converted, blueprint):
    return jax.tree_multimap(lambda leaf, do_restore: DIST_ID[leaf] if do_restore else leaf, converted, blueprint)

def _flatten_dist(dist: npd.Distribution):
    params, aux = dist.tree_flatten()
    return params, *_auxiliary_to_jax_type(aux), DIST_ID.index(dist.__class__)

def vmap_distribution(f: Callable[Any, npd.Distribution], in_axes=0, out_axes=0, axis_name=None):
    """Helper function that vmaps a function that may return a distribution as output. """
    flat_f = lambda *args, **kwargs: _flatten_dist(f(*args, **kwargs))  # this returns the flat_dist, type tuple
    vmapped = jax.vmap(flat_f, in_axes=in_axes, out_axes=(out_axes, None, None, None), axis_name=axis_name)

    def unflattened(*args, **kwargs):
        vmapped_output, aux_converted, aux_blueprint, cls_id = vmapped(*args, **kwargs)
        cls = DIST_ID[cls_id]
        aux = _restore_auxiliary(aux_converted, aux_blueprint)
        return cls.tree_unflatten(aux, vmapped_output)

    return unflattened

if __name__ == "__main__":
    import jax

    # def get_dist(mean):
    #     return DiagonalGaussian(mean, jnp.ones_like(mean), mean.ndim)

    def get_dist(mean):
        return Independent(Normal(mean, jnp.ones_like(mean)), 2)

    get_dists = vmap_distribution(get_dist)

    key = jax.random.PRNGKey(42)
    mean = jax.random.normal(key, (8, 1))
    means = jax.random.normal(key, (3, 8, 1))

    simple_dist = get_dist(mean)  ##  .get_numpyro_distribution()
    batched_dist = get_dists(means)  ## .get_numpyro_distribution()

    print(simple_dist.batch_shape, "&", simple_dist.event_shape)  # prints: () & (8,1)
    print(batched_dist.batch_shape, "&", batched_dist.event_shape)  # Should print: (3,) & (8,1)

    assert np.arra