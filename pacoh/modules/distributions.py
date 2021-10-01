import warnings

import numpyro.distributions
from numpyro.distributions import Distribution, Normal
from numpyro.distributions import TransformedDistribution
from numpyro.distributions.transforms import AffineTransform
import torch
import numpy as np
from jax import numpy as jnp
import haiku as hk

from pacoh.modules.common import PositiveParameter


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
        # self.loc_tensor = torch.tensor(normalization_mean).float().reshape((1,))
        self.scale = normalization_std.reshape((1,))
        normalization_transform = AffineTransform(loc=normalization_mean, scale=normalization_std)
        super().__init__(base_dist, normalization_transform)

    @property
    def mean(self):
        return self.transforms[0](self.base_dist.mean)

    @property
    def stddev(self):
        return np.exp(np.log(self.base_dist.scale) + np.log(self.scale))

    @property
    def variance(self):
        return np.exp(np.log(self.base_dist.variance) + 2 * np.log(self.scale))


class JAXGaussianLikelihood(hk.Module):
    # TODO handle output dim => no need for expand_dims below
    def __init__(self, variance: float = 1.0, variance_constraint_gt=0.0):
        super().__init__()
        # TODO, to avoid numerical errors, I should store the std, not the variance, no
        self.variance = PositiveParameter(variance, boundary_value=variance_constraint_gt)

    def __call__(self, posterior):
        scale = jnp.sqrt(posterior.scale**2 + self.variance())
        return Normal(loc=posterior.loc, scale=scale)

    def log_prob(self, ys_true, ys_pred):
        scale = jnp.sqrt(self.variance())
        scale = jnp.expand_dims(scale, 0)
        logprob = Normal(scale=scale).log_prob(ys_true - ys_pred)
        return logprob

    def get_posterior_from_means(self, loc):
        batch_size = loc.shape[0]
        stds = jnp.repeat(jnp.sqrt(self.variance()), batch_size, axis=0)
        return Normal(loc, scale=stds)
