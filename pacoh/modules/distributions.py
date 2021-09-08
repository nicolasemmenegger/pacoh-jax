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
    def __init__(self, variance: float = 1.0, variance_constraint_gt=0.0):
        super().__init__()
        self.variance = PositiveParameter(variance, boundary_value=variance_constraint_gt)

    def __call__(self, posterior):
        scale = jnp.sqrt(posterior.scale**2 + self.variance())
        return Normal(loc=posterior.loc, scale=scale)
