from numpyro.distributions import Distribution, Normal
from numpyro.distributions import TransformedDistribution
from numpyro.distributions.transforms import AffineTransform
import torch
import numpy as np
from jax import numpy as jnp


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


class UnnormalizedExpDist(Distribution):
    r"""
    Creates a an unnormalized distribution with density function with
    density proportional to exp(exponent_fn(value))

    Args:
      exponent_fn: callable that outputs the exponent
    """

    def __init__(self, exponent_fn):
        self.exponent_fn = exponent_fn
        super().__init__()

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        return self.exponent_fn(value)


class FactorizedNormal(Distribution):

    def __init__(self, loc, scale, summation_axis=-1):
        self.normal_dist = torch.distributions.Normal(loc, scale)
        self.summation_axis = summation_axis

    def log_prob(self, value):
        return torch.sum(self.normal_dist.log_prob(value), dim=self.summation_axis)


class EqualWeightedMixtureDist(Distribution):

    def __init__(self, dists, batched=False, num_dists=None):
        self.batched = batched
        if batched:
            assert isinstance(dists, torch.distributions.Distribution)
            self.num_dists = dists.batch_shape if num_dists is None else num_dists
            event_shape = dists.event_shape
        else:
            assert all([isinstance(d, torch.distributions.Distribution) for d in dists])
            event_shape = dists[0].event_shape
            self.num_dists = len(dists)
        self.dists = dists

        super().__init__(event_shape=event_shape)

    @property
    def mean(self):
        if self.batched:
            return torch.mean(self.dists.mean, dim=0)
        else:
            return torch.mean(torch.stack([dist.mean for dist in self.dists], dim=0), dim=0)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    @property
    def variance(self):
        if self.batched:
            means = self.dists.mean
            vars = self.dists.variance
        else:
            means = torch.stack([dist.mean for dist in self.dists], dim=0)
            vars = torch.stack([dist.variance for dist in self.dists], dim=0)

        var1 = torch.mean((means - torch.mean(means, dim=0)) ** 2, dim=0)
        var2 = torch.mean(vars, dim=0)

        # check shape
        assert var1.shape == var2.shape
        return var1 + var2

    @property
    def arg_constraints(self):
        return {}

    def log_prob(self, value):
        if self.batched:
            log_probs_dists = self.dists.log_prob(value)
        else:
            log_probs_dists = torch.stack([dist.log_prob(value) for dist in self.dists])
        return torch.logsumexp(log_probs_dists, dim=0) - torch.log(torch.tensor(self.num_dists).float())

    def cdf(self, value):
        if self.batched:
            cum_p = self.dists.cdf(value)
        else:
            cum_p = torch.stack([dist.cdf(value) for dist in self.dists])
        assert cum_p.shape[0] == self.num_dists
        return torch.mean(cum_p, dim=0)

    def icdf(self, quantile):
        left = - 1e8 * torch.ones(quantile.shape)
        right = + 1e8 * torch.ones(quantile.shape)
        fun = lambda x: self.cdf(x) - quantile
        return find_root_by_bounding(fun, left, right)


class CatDist(Distribution):

    def __init__(self, dists, reduce_event_dim=True):
        assert all([len(dist.event_shape) == 1 for dist in dists])
        assert all([len(dist.batch_shape) == 0 for dist in dists])
        self.reduce_event_dim = reduce_event_dim
        self.dists = dists
        self._event_shape = torch.Size((sum([dist.event_shape[0] for dist in self.dists]),))

    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='sample')

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='rsample')

    def log_prob(self, value):
        idx = 0
        log_probs = []
        for dist in self.dists:
            n = dist.event_shape[0]
            if value.ndim == 1:
                val = value[idx:idx + n]
            elif value.ndim == 2:
                val = value[:, idx:idx + n]
            elif value.ndim == 2:
                val = value[:, :, idx:idx + n]
            else:
                raise NotImplementedError('Can only handle values up to 3 dimensions')
            log_probs.append(dist.log_prob(val))
            idx += n

        for i in range(len(log_probs)):
            if log_probs[i].ndim == 0:
                log_probs[i] = log_probs[i].reshape((1,))

        if self.reduce_event_dim:
            return torch.sum(torch.stack(log_probs, dim=0), dim=0)
        return torch.stack(log_probs, dim=0)

    def _sample(self, sample_shape, sample_fn='sample'):
        return torch.cat([getattr(d, sample_fn)(sample_shape) for d in self.dists], dim=-1)


class JAXGaussianLikelihood:
    def __init__(self, variance: float = 1.0):
        self.variance = variance

    def __call__(self, posterior):

        scale = jnp.sqrt(posterior.scale**2 + self.variance)
        return Normal(loc=posterior.loc, scale=scale)
