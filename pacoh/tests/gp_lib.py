import unittest

import haiku as hk
import numpy as np
from jax import numpy as jnp

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



if __name__ == '__main__':
    unittest.main()
