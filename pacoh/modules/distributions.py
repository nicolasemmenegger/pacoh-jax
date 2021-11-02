import numpyro
from numpyro.distributions import Normal
from numpyro.distributions import TransformedDistribution
from numpyro.distributions.transforms import AffineTransform
import numpy as np
from jax import numpy as jnp
import haiku as hk

from pacoh.modules.common import PositiveParameter
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
        # TODO find out if I broke something here
        # self.norm_scale = normalization_std.reshape((1,)) # so we can apply it on batches
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
        super().__init__()
        variance = jnp.ones((output_dim,))*variance
        if learn_likelihood:
            self.variance = PositiveParameter(variance, boundary_value=variance_constraint_gt)
        else:
            self.variance = lambda: variance

    def __call__(self, posterior):
        scale = jnp.sqrt(posterior.scale**2 + self.variance())
        return Normal(loc=posterior.loc, scale=scale)

    def log_prob(self, ys_true, ys_pred):
        scale = jnp.sqrt(self.variance())
        logprob = Normal(scale=scale).log_prob(ys_true - ys_pred) # log likelihood of the data under the modeled variance
        return logprob

    def get_posterior_from_means(self, loc):
        batch_size = loc.shape[0]
        var = self.variance()
        stds = jnp.broadcast_to(jnp.sqrt(var), (batch_size, *var.shape))
        return Normal(loc, scale=stds)


def get_mixture(pred_dists, n):
    pred_dists = stack_distributions(pred_dists)
    mixture_weights = numpyro.distributions.Categorical(probs=jnp.ones((n,)) / n)
    return numpyro.distributions.MixtureSameFamily(mixture_weights, pred_dists)