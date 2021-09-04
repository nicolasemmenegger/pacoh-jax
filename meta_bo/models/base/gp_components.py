import functools
import warnings
from typing import Optional

import numpyro.distributions
from gpytorch.functions import RBFCovariance
from gpytorch.utils.broadcasting import _mul_broadcast_shape
import gpytorch
import torch
from abc import ABC, abstractmethod
from jax import numpy as jnp, vmap
from jax.scipy.linalg import cho_solve, cho_factor
import haiku as hk

from meta_bo.models.base.common import PositiveParameter
from meta_bo.models.base.neural_network import JAXNeuralNetwork


class ConstantMeanLight(gpytorch.means.Mean):
    def __init__(self, constant=torch.ones(1), batch_shape=torch.Size()):
        super(ConstantMeanLight, self).__init__()
        self.batch_shape = batch_shape
        self.constant = constant

    def forward(self, input):
        if input.shape[:-2] == self.batch_shape:
            return self.constant.expand(input.shape[:-1])
        else:
            return self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))


class SEKernelLight(gpytorch.kernels.Kernel):

    def __init__(self, lengthscale=torch.tensor([1.0]), output_scale=torch.tensor(1.0)):
        super(SEKernelLight, self).__init__(batch_shape=(lengthscale.shape[0],))
        self.length_scale = lengthscale
        self.ard_num_dims = lengthscale.shape[-1]
        self.output_scale = output_scale
        self.postprocess_rbf = lambda dist_mat: self.output_scale * dist_mat.div_(-2).exp_()

    def forward(self, x1, x2, diag=False, **params):
        if (
                x1.requires_grad
                or x2.requires_grad
                or (self.ard_num_dims is not None and self.ard_num_dims > 1)
                or diag
        ):
            x1_ = x1.div(self.length_scale)
            x2_ = x2.div(self.length_scale)
            return self.covar_dist(x1_, x2_, square_dist=True, diag=diag,
                                   dist_postprocess_func=self.postprocess_rbf,
                                   postprocess=True, **params)
        return self.output_scale * RBFCovariance().apply(x1, x2, self.length_scale,
                                                         lambda x1, x2: self.covar_dist(x1, x2,
                                                                                        square_dist=True,
                                                                                        diag=False,
                                                                                        dist_postprocess_func=self.postprocess_rbf,
                                                                                        postprocess=False,
                                                                                        **params))

class HomoskedasticNoiseLight(gpytorch.likelihoods.noise_models._HomoskedasticNoiseBase):

    def __init__(self, noise_var, *params, **kwargs):
        self.noise_var = noise_var
        self._modules = {}
        self._parameters = {}

    @property
    def noise(self):
        return self.noise_var

    @noise.setter
    def noise(self, value):
        self.noise_var = value


class GaussianLikelihoodLight(gpytorch.likelihoods._GaussianLikelihoodBase):

    def __init__(self, noise_var, batch_shape=torch.Size()):
        self.batch_shape = batch_shape
        self._modules = {}
        self._parameters = {}

        noise_covar = HomoskedasticNoiseLight(noise_var)
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.noise = value

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise_covar.noise

        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        return res.mul(-0.5).sum(-1)


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
        hk.set_state("xs", None)
        hk.set_state("ys", None)

    def _ys_centered(self, xs, ys):
        return ys - self.mean_set(xs).flatten()

    def _data_cov_with_noise(self, xs):
        data_cov = self.cov_set_set(xs, xs)
        return data_cov + jnp.eye(*data_cov.shape) * self.noise_variance()

    def fit(self, xs: jnp.ndarray, ys: jnp.ndarray):
        hk.set_state("xs", xs)
        hk.set_state("ys", ys)
        data_cov_w_noise = self._data_cov_with_noise(xs)
        hk.set_state("cholesky", cho_factor(data_cov_w_noise, lower=True))

    @functools.partial(vmap, in_axes=(None, 0))
    def posterior(self, x: jnp.ndarray):
        """prediction without noise"""
        xs = hk.get_state("xs")
        ys = hk.get_state("ys")
        if xs is not None:
            # we have data
            new_cov_row = self.cov_vec_set(x, xs)
            mean = self.mean_module(x) + jnp.dot(new_cov_row, cho_solve(hk.get_state("cholesky"), self._ys_centered(xs, ys)))
            var = self.cov_vec_vec(x, x) - jnp.dot(new_cov_row, cho_solve(hk.get_state("cholesky"), new_cov_row))
            std = jnp.sqrt(var)
            return numpyro.distributions.Normal(loc=mean, scale=stddev)
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

    def marginal_ll(self):
        # the following computes the log likelihood of the training data, i.e. does not use the arguments
        # How do I differentiate here?
        warnings.warn("I do not think that this is implemented correctly, as I don't differentiate w.r.t. to the parameter"
                      "of the likelihood variance that is inside the cholesky decomposition. I think I need to have some kind of "
                      "option to keep the cholesky or not, depending on if I am meta-training or target-training. In the former case, I need to "
                      "differentiate through the colesky decomp, whereas in the latter I don't. ")
        xs = hk.get_state("xs")
        ys = hk.get_state("ys")
        ys_centered = self._ys_centered(xs, ys)
        solved = cho_solve(hk.get_state("cholesky"), ys_centered)
        data_cov = self._data_cov_with_noise(xs)
        ll = -0.5 * jnp.dot(ys_centered, solved)
        ll += -0.5 * jnp.linalg.slogdet(data_cov)[1]
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
