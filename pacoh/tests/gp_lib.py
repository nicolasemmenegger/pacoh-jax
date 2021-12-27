import unittest

import haiku as hk
import numpy as np
import numpyro
from numpyro.distributions import Independent, MultivariateNormal, Normal
from jax import numpy as jnp

from pacoh.modules.distributions import multivariate_kl
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
                    module = JAXConstantMean(output_dim=output_dim, initial_constant=jnp.ones((output_dim,)) * 3.14)
                    return module(xs)

                num_samps = 10
                fwds = [forward, const_fwd, const_fwd_two]
                expected = [jnp.zeros((num_samps, output_dim))] + [jnp.ones((num_samps, output_dim))*3.14]*2

                for fwd, expectation in zip(fwds, expected):
                    init, apply = hk.transform(fwd)
                    dummy_input = jnp.ones((num_samps, input_dim))
                    params = init(None, dummy_input)
                    res = apply(params, None, dummy_input)
                    self.assertEqual(res.shape, (num_samps, output_dim)) # check shapes
                    np.testing.assert_array_equal(res, expectation)

                    # it should also work for a single point
                    dummy_input = jnp.ones((input_dim,))
                    res = apply(params, None, dummy_input)
                    self.assertEqual(res.shape, (output_dim,))  # check shapes
                    np.testing.assert_array_equal(res, expectation[0])

    def test_kernel_interfaces(self):
        pass


class TestAuxiliaryStuff(unittest.TestCase):
    def test_diagonalcase(self):
        loc1 = jnp.zeros((3,))
        loc2 = 0.1 + jnp.zeros((3,))
        dcov1 = 1.1 * jnp.ones((3,))
        dcov2 = 0.9 * jnp.ones((3,))

        # diagonal gaussian
        dist1 = numpyro.distributions.Normal(loc=loc1, scale=dcov1)
        dist2 = numpyro.distributions.Normal(loc=loc2, scale=dcov2)
        dist1 = numpyro.distributions.Independent(dist1, 1)
        dist2 = numpyro.distributions.Independent(dist2, 1)
        diagonalcase = numpyro.distributions.kl_divergence(dist1, dist2)

        # full gaussian
        cov1 = jnp.diag(dcov1)
        cov2 = jnp.diag(dcov2)
        fdist1 = MultivariateNormal(loc1, cov1)
        fdist2 = MultivariateNormal(loc2, cov2)
        mykl = multivariate_kl(fdist1, fdist2)

        print(diagonalcase)
        print(mykl)

        self.assertAlmostEquals(diagonalcase, mykl)

    def compare_to_pytorch(self):
        pass


if __name__ == '__main__':
    unittest.main()
