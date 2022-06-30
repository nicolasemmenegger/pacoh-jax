import unittest

import haiku as hk
import jax.random
import numpy as np
import numpyro
import torch
from numpyro.distributions import MultivariateNormal
from jax import numpy as jnp

from pacoh.util.distributions import multivariate_kl
from pacoh.modules.means import JAXConstantMean, JAXZeroMean


class TestMeanAndKernels(unittest.TestCase):
    def test_means(self):
        for input_dim in range(1, 4):
            for output_dim in range(1, 4):

                def forward(xs):
                    module = JAXZeroMean(output_dim=output_dim)
                    return module(xs)

                def const_fwd(xs):
                    module = JAXConstantMean(output_dim=output_dim, initial_constant=3.14)
                    return module(xs)

                def const_fwd_two(xs):
                    module = JAXConstantMean(
                        output_dim=output_dim,
                        initial_constant=jnp.ones((output_dim,)) * 3.14,
                    )
                    return module(xs)

                num_samps = 10
                fwds = [forward, const_fwd, const_fwd_two]
                expected = [jnp.zeros((num_samps, output_dim))] + [
                    jnp.ones((num_samps, output_dim)) * 3.14
                ] * 2

                for fwd, expectation in zip(fwds, expected):
                    init, apply = hk.transform(fwd)
                    dummy_input = jnp.ones((num_samps, input_dim))
                    params = init(None, dummy_input)
                    res = apply(params, None, dummy_input)
                    self.assertEqual(res.shape, (num_samps, output_dim))  # check shapes
                    np.testing.assert_array_equal(res, expectation)

                    # it should also work for a single point
                    dummy_input = jnp.ones((input_dim,))
                    res = apply(params, None, dummy_input)
                    self.assertEqual(res.shape, (output_dim,))  # check shapes
                    np.testing.assert_array_equal(res, expectation[0])

    def test_kernel_interfaces(self):
        pass


class TestAuxiliaryStuff(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        loc1 = jnp.zeros((3,))
        loc2 = 0.1 + jnp.zeros((3,))
        dcov1 = 1.1 * jnp.ones((3,))
        dcov2 = 0.9 * jnp.ones((3,))

        # diagonal gaussian
        dist1 = numpyro.distributions.Normal(loc=loc1, scale=jnp.sqrt(dcov1))
        dist2 = numpyro.distributions.Normal(loc=loc2, scale=jnp.sqrt(dcov2))
        self.dist1 = numpyro.distributions.Independent(dist1, 1)
        self.dist2 = numpyro.distributions.Independent(dist2, 1)

        # full gaussian
        cov1 = jnp.diag(dcov1)
        cov2 = jnp.diag(dcov2)
        self.fdist1 = MultivariateNormal(loc1, cov1)
        self.fdist2 = MultivariateNormal(loc2, cov2)

        cov1 = torch.from_numpy(np.array(cov1))
        cov2 = torch.from_numpy(np.array(cov2))

        self.np1 = torch.distributions.MultivariateNormal(torch.from_numpy(np.array(loc1)), cov1)
        self.np2 = torch.distributions.MultivariateNormal(torch.from_numpy(np.array(loc2)), cov2)

    def test_diagonal_case(self):
        diagonalcase = numpyro.distributions.kl_divergence(self.dist1, self.dist2)
        mykl = multivariate_kl(self.fdist1, self.fdist2)
        torchkl = torch.distributions.kl_divergence(self.np1, self.np2)

        self.assertAlmostEqual(diagonalcase, mykl, places=5)
        self.assertAlmostEqual(torchkl.numpy(), mykl, places=5)

    def test_full_covariance(self):
        rng = jax.random.PRNGKey(0)
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        cov1 = jax.random.normal(r1, (5, 5))
        cov2 = jax.random.normal(r2, (5, 5))
        cov1 = cov1.T @ cov1
        cov2 = cov2.T @ cov2
        l1 = jax.random.normal(r3, (5,))
        l2 = jax.random.normal(r4, (5,))

        npd1 = MultivariateNormal(l1, cov1)
        npd2 = MultivariateNormal(l2, cov2)

        torchd1 = torch.distributions.MultivariateNormal(
            self.jnp_to_torch_tensor(l1), self.jnp_to_torch_tensor(cov1)
        )
        torchd2 = torch.distributions.MultivariateNormal(
            self.jnp_to_torch_tensor(l2), self.jnp_to_torch_tensor(cov2)
        )

        mykl = multivariate_kl(npd1, npd2)
        torchkl = torch.distributions.kl_divergence(torchd1, torchd2)

        self.assertAlmostEqual(torchkl.numpy(), mykl, places=3)

    @staticmethod
    def jnp_to_torch_tensor(arr):
        return torch.from_numpy(np.array(arr))


class TestExactGP(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
