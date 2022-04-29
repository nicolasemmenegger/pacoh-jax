import unittest

import jax.random
import numpy as np
import numpyro.distributions
import torch.distributions
from jax import numpy as jnp

from pacoh.models.f_pacoh_map import F_PACOH_MAP_GP
from pacoh.modules.domain import ContinuousDomain
from pacoh.modules.kernels import rbf_cov
from pacoh.util.distributions import multivariate_kl


class TestFunctionalKL(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hyperprior_lengthscale = 1.0
        self.hyperprior_outputscale = 1.3
        self.hyperprior_noise_std = 1.0
        self.f_pacoh_map = F_PACOH_MAP_GP(
            1,
            1,
            ContinuousDomain(jnp.ones((1,)) * -6, jnp.ones((1,)) * 6),
            learning_mode="both",
            weight_decay=0.5,
            task_batch_size=5,
            num_tasks=5,
            covar_module="NN",
            mean_module="NN",
            mean_nn_layers=(32, 32),
            feature_dim=2,
            kernel_nn_layers=(32, 32),
            hyperprior_lengthscale=self.hyperprior_lengthscale,
            hyperprior_outputscale=self.hyperprior_outputscale,
            hyperprior_noise_var=self.hyperprior_noise_std,
        )

    def test_hyperprior_covariance(self):
        n = 10
        xs = jax.random.normal(jax.random.PRNGKey(23), (n, 1))
        prior = self.f_pacoh_map.hyperprior_marginal(xs)
        ls = lambda: self.hyperprior_lengthscale
        os = lambda: self.hyperprior_outputscale

        covar = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov = rbf_cov(xs[i], xs[j], ls, os)
                covar[i, j] = cov

        np.testing.assert_array_equal(covar, np.array(prior.covariance_matrix))

    def test_multivariate_marginal_kl(self):
        n = 20
        cov1 = jax.random.normal(jax.random.PRNGKey(32), (n, n))
        cov1 = cov1.T @ cov1 + 0.1 * jnp.eye(n)
        mean1 = jax.random.normal(jax.random.PRNGKey(13), (n,))
        cov2 = jax.random.normal(jax.random.PRNGKey(53), (n, n))
        cov2 = cov2.T @ cov2 + 0.1 * jnp.eye(n)
        mean2 = jax.random.normal(jax.random.PRNGKey(24), (n,))

        npd1 = numpyro.distributions.MultivariateNormal(mean1, cov1)
        npd2 = numpyro.distributions.MultivariateNormal(mean2, cov2)

        td1 = torch.distributions.MultivariateNormal(
            torch.from_numpy(np.array(mean1)), torch.from_numpy(np.array(cov1))
        )
        td2 = torch.distributions.MultivariateNormal(
            torch.from_numpy(np.array(mean2)), torch.from_numpy(np.array(cov2))
        )
        my_kl = multivariate_kl(npd1, npd2)
        torch_kl = torch.distributions.kl_divergence(td1, td2)

        print(my_kl)
        print(torch_kl)
