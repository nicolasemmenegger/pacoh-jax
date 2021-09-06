import functools
import numpyro.distributions
from abc import ABC, abstractmethod
from jax import numpy as jnp, vmap
from jax.scipy.linalg import cho_solve, cho_factor
import haiku as hk

class JAXExactGP:
    """
        A simple implementation of a gaussian process module with exact inference and Gaussian Likelihood
    """
    def __init__(self, mean_module, cov_module, likelihood):
        """
        Args:
            mean_module: hk.Module
            cov_module: hk.Module
            likelihood: hk.Module

        :param mean_module:
        :param cov_module:
        :param likelihood:
        """
        self.mean_module = mean_module
        self.noise_variance = likelihood.variance
        self.likelihood = likelihood

        self.cov_vec_vec = cov_module
        self.cov_vec_set = vmap(cov_module.__call__, (None, 0))
        self.cov_set_set = vmap(self.cov_vec_set.__call__, (0, None))
        self.mean_set = vmap(self.mean_module.__call__)
        hk.set_state("xs", jnp.zeros((0, 1), dtype=jnp.float32))
        hk.set_state("ys", jnp.zeros((0,), dtype=jnp.float32))


    def _ys_centered(self, xs, ys):
        return ys - self.mean_set(xs).flatten()

    def _data_cov_with_noise(self, xs):
        data_cov = self.cov_set_set(xs, xs)
        return data_cov + jnp.eye(*data_cov.shape) * self.noise_variance()

    def fit(self, xs: jnp.ndarray, ys: jnp.ndarray) -> None:
        """ Fit adds the data in any case, stores cholesky if we are at eval stage for later reuse"""
        hk.set_state("xs", xs)
        hk.set_state("ys", ys)
        data_cov_w_noise = self._data_cov_with_noise(xs)
        hk.set_state("cholesky", cho_factor(data_cov_w_noise, lower=True))

    @functools.partial(vmap, in_axes=(None, 0))
    def posterior(self, x: jnp.ndarray):
        xs = hk.get_state("xs")
        ys = hk.get_state("ys")
        if xs.size > 0:
            # we have data
            new_cov_row = self.cov_vec_set(x, xs)
            mean = self.mean_module(x) + jnp.dot(new_cov_row, cho_solve(hk.get_state("cholesky"), self._ys_centered(xs, ys)))
            var = self.cov_vec_vec(x, x) - jnp.dot(new_cov_row, cho_solve(hk.get_state("cholesky"), new_cov_row))
            std = jnp.sqrt(var)
            return numpyro.distributions.Normal(loc=mean, scale=std)
        else:
            return self._prior(x)

    def pred_dist(self, x, noiseless=False):
        """prediction with noise"""
        predictive_dist_noiseless = self.posterior(x)
        return self.likelihood(predictive_dist_noiseless)

    @functools.partial(vmap, in_axes=(None, 0))
    def prior(self, x):
        return self._prior(x)

    def _prior(self, x):
        mean = self.mean_module(x)
        stddev = jnp.sqrt(self.cov_vec_vec(x, x))
        return numpyro.distributions.Normal(loc=mean, scale=stddev)

    def add_data_and_refit(self, xs, ys):
        old_xs = hk.get_state("xs")
        old_ys = hk.get_state("ys")
        all_xs = jnp.concatenate([old_xs, xs])
        all_ys = jnp.concatenate([old_ys, ys])
        self.fit(all_xs, all_ys)

    """ ---- training utilities ---- """
    def marginal_ll(self, xs, ys):
        # computes the marginal log-likelihood of ys given xs and a posterior
        # computed on (xs,ys). This is differentiable and uses no state
        ys_centered = self._ys_centered(xs, ys)
        data_cov_w_noise = self._data_cov_with_noise(xs)
        cholesky = cho_factor(data_cov_w_noise, lower=True)
        solved = cho_solve(cholesky, ys_centered)
        ll = -0.5 * jnp.dot(ys_centered, solved)
        ll += -0.5 * jnp.linalg.slogdet(data_cov_w_noise)[1]
        ll -= xs.shape[0] / 2 * jnp.log(2 * jnp.pi)
        return ll

class JAXMean(hk.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, x):
        pass


class JAXConstantMean(JAXMean):
    """Mean module with a learnable mean. """
    def __init__(self, initial_constant=0.0):
        super().__init__()
        self.init_constant = initial_constant

    def __call__(self, x):
        # works for both batch or unbatched
        mean = hk.get_parameter("mu", shape=[], dtype=jnp.float64,
                                init=hk.initializers.Constant(self.init_constant))
        return jnp.ones(x.shape, dtype=jnp.float64) * mean


class JAXZeroMean(JAXMean):
    """Always zero, not learnable. """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return jnp.zeros(x.shape)


#
# from gpytorch.models.approximate_gp import ApproximateGP
# from gpytorch.variational import CholeskyVariationalDistribution
# from gpytorch.variational import VariationalStrategy
#
#
# class LearnedGPRegressionModelApproximate(ApproximateGP):
#     """GP model which can take a learned mean and learned kernel function."""
#
#     def __init__(self, train_x, train_y, likelihood, learned_kernel=None, learned_mean=None, mean_module=None,
#                  covar_module=None, beta=1.0):
#
#         self.beta = beta
#         self.n_train_samples = train_x.shape[0]
#
#         variational_distribution = CholeskyVariationalDistribution(self.n_train_samples)
#         variational_strategy = VariationalStrategy(self, train_x, variational_distribution,
#                                                    learn_inducing_locations=False)
#         super().__init__(variational_strategy)
#
#         if mean_module is None:
#             self.mean_module = gpytorch.means.ZeroMean()
#         else:
#             self.mean_module = mean_module
#
#         self.covar_module = covar_module
#
#         self.learned_kernel = learned_kernel
#         self.learned_mean = learned_mean
#         self.likelihood = likelihood
#
#     def forward(self, x):
#         # feed through kernel NN
#         if self.learned_kernel is not None:
#             projected_x = self.learned_kernel(x)
#         else:
#             projected_x = x
#
#         # feed through mean module
#         if self.learned_mean is not None:
#             mean_x = self.learned_mean(x).squeeze()
#         else:
#             mean_x = self.mean_module(projected_x).squeeze()
#
#         covar_x = self.covar_module(projected_x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
#     def prior(self, x):
#         return self.forward(x)
#
#     def kl(self):
#         return self.variational_strategy.kl_divergence()
#
#     def pred_dist(self, x):
#         self.eval()
#         return self.likelihood(self.__call__(x))
#
#     def pred_ll(self, x, y):
#         variational_dist_f = self.__call__(x)
#         return self.likelihood.expected_log_prob(y, variational_dist_f).sum(-1)
#
#     @property
#     def variational_distribution(self):
#         return self.variational_strategy._variational_distribution

from gpytorch.models import ExactGP