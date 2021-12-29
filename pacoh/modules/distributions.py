import warnings

import numpyro
from numpyro.distributions import Normal, TransformedDistribution, MultivariateNormal, Independent
from numpyro.distributions.transforms import AffineTransform
import numpy as np
from jax import numpy as jnp
import haiku as hk
from jax.scipy.linalg import cho_solve, cho_factor


from pacoh.modules.common import PositiveParameter
from pacoh.util.constants import LIKELIHOOD_MODULE_NAME
from pacoh.util.tree import stack_distributions


class AffineTransformedDistribution(TransformedDistribution):
    r"""
    Implements an affine transformation of a probability distribution p(x)

    x_transformed = mean + std * x , x \sim p(x)

    Args:
        base_dist: (torch.distributions.Distribution) probability distribution to transform
        normalization_mean: (np.ndarray) additive factor to add to x
        normalization_std: (np.ndarray) multiplicative factor for scaling x
    """

    def __init__(self, base_dist, normalization_mean, normalization_std):
        self.norm_scale = normalization_std
        normalization_transform = AffineTransform(loc=normalization_mean, scale=normalization_std)
        super().__init__(base_dist, normalization_transform)

    @property
    def mean(self):
        return self.transforms[0](self.base_dist.mean)

    @property
    def stddev(self):
        if hasattr(self.base_dist, "scale"):
            return np.exp(np.log(self.base_dist.scale) + np.log(self.norm_scale))
        elif hasattr(self.base_dist, "variance"):
            return np.exp(0.5*np.log(self.base_dist.variance) + np.log(self.norm_scale))
        elif hasattr(self.base_dist, "stddev"):
            return np.exp(np.log(self.base_dist.stddev) + np.log(self.norm_scale))

    @property
    def variance(self):
        return np.exp(np.log(self.base_dist.variance) + 2 * np.log(self.norm_scale))

    @property
    def iid_normal(self):
        return Normal(loc=self.mean, scale=self.stddev)


class JAXGaussianLikelihood(hk.Module):
    def __init__(self, variance: float = 1.0, variance_constraint_gt=0.0, output_dim=1, learn_likelihood=True):
        super().__init__(LIKELIHOOD_MODULE_NAME)
        variance = jnp.ones((output_dim,))*variance
        if learn_likelihood:
            self.variance = PositiveParameter(variance, boundary_value=variance_constraint_gt)
        else:
            self.variance = lambda: variance

    def __call__(self, posterior):
        d = posterior.loc.shape[0]
        if isinstance(posterior, numpyro.distributions.MultivariateNormal):
            cov_with_noise = posterior.covariance_matrix + jnp.eye(d, d) * self.variance()
            return numpyro.distributions.MultivariateNormal(loc=posterior.loc, covariance_matrix=cov_with_noise)
        elif isinstance(posterior, numpyro.distributions.Independent):
            scale = jnp.sqrt(posterior.variance + self.variance())
            return get_diagonal_gaussian(posterior.loc, scale)
        else:
            raise ValueError("posterior should be either a multivariate diagonal or full covariance gaussian")

    # def log_prob(self, ys_true, ys_pred):
    #     scale = jnp.sqrt(self.variance())
    #     logprob = Normal(scale=scale).log_prob(ys_true - ys_pred)  # log likelihood of the data under the modeled variance
    #     return logprob

    def get_posterior_from_means(self, loc):
        raise NotImplementedError("I think this should be redone")
        batch_size = loc.shape[0]
        var = self.variance()
        stds = jnp.broadcast_to(jnp.sqrt(var), (batch_size, *var.shape))
        return Normal(loc, scale=stds)


def get_mixture(pred_dists, n):
    pred_dists = stack_distributions(pred_dists)
    mixture_weights = numpyro.distributions.Categorical(probs=jnp.ones((n,)) / n)
    return numpyro.distributions.MixtureSameFamily(mixture_weights, pred_dists)


def get_diagonal_gaussian(loc, scale):
    assert loc.shape == scale.shape and loc.ndim <= 2
    return numpyro.distributions.Independent(numpyro.distributions.Normal(loc, scale), loc.ndim)


def multivariate_kl(dist1: MultivariateNormal, dist2: MultivariateNormal) -> float:
    # TODO this could probably be sped up a bit by combining certain cholesky factorizations
    logdets = jnp.linalg.slogdet(dist2.covariance_matrix)[1] - jnp.linalg.slogdet(dist1.covariance_matrix)[1]
    d2chol = cho_factor(dist2.covariance_matrix)
    trace = jnp.trace(cho_solve(d2chol, dist1.covariance_matrix))
    locdiff = dist2.loc - dist1.loc
    bilin = locdiff @ cho_solve(d2chol, locdiff)

    kl = logdets - dist1.loc.shape[0] + trace + bilin
    return 0.5*kl


def diagonalize_gaussian(dist):
    if isinstance(dist, Independent):
        return dist
    elif isinstance(dist, MultivariateNormal):
        return get_diagonal_gaussian(dist.mean, jnp.sqrt(jnp.diag(dist.covariance_matrix)))
    else:
        raise ValueError("unknown argument type")