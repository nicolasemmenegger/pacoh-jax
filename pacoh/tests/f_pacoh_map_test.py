import unittest

import jax.random
import numpy as np
from jax import numpy as jnp

from pacoh.models.f_pacoh_map import F_PACOH_MAP_GP
from pacoh.modules.domain import ContinuousDomain
from pacoh.modules.kernels import rbf_cov


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
        # this test is pretty useless now, but I have found a bug thanks to it
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

        print(covar)
        np.testing.assert_array_equal(covar, np.array(prior.covariance_matrix))
