import functools
import warnings
from typing import Callable, Optional

import haiku as hk
import jax

from pacoh.modules.common import PositiveParameter
from jax import numpy as jnp


def rbf_cov(x1, x2, ls_param, os_param):
    return os_param() * jnp.exp(-0.5 * (jnp.sum(jax.lax.square(x1 - x2))) / (ls_param() ** 2))

class JAXKernel(hk.Module):
    pass

class JAXRBFKernel(JAXKernel):
    def __init__(self,
                 input_dim,
                 length_scale=1.0,
                 output_scale=1.0,
                 length_scale_constraint_gt=0.0,
                 output_scale_constraint_gt=0.0):
        super().__init__()
        self.input_dim = input_dim

        self.output_scale = PositiveParameter(initial_value=output_scale,
                                              boundary_value=output_scale_constraint_gt,
                                              name="OutputScale")
        self.length_scale = PositiveParameter(initial_value=length_scale,
                                              boundary_value=length_scale_constraint_gt,
                                              name="LengthScale")

    def __call__(self, x1, x2):
        return rbf_cov(x1, x2, self.length_scale, self.output_scale)


class JAXRBFKernelNN(JAXKernel):
    def __init__(self,
                 input_dim,
                 feature_dim,
                 layer_sizes=(64, 64),
                 length_scale=1.0,
                 output_scale=1.0,
                 length_scale_constraint_gt=0.0,
                 output_scale_constraint_gt=0.0):

        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.output_scale = PositiveParameter(initial_value=output_scale,
                                              boundary_value=output_scale_constraint_gt,
                                              name="OutputScale")
        self.length_scale = PositiveParameter(initial_value=length_scale,
                                              boundary_value=length_scale_constraint_gt,
                                              name="LengthScale")

        self.nn_ftr_map = hk.nets.MLP(output_sizes=(1,), activation=lambda x: x)
        self.dummy_param = hk.get_parameter("stupid", dtype=jnp.float32, shape=[], init=hk.initializers.Constant(0.0))

    def __call__(self, x1, x2=None):
        x1 = self.nn_ftr_map(x1)
        x2 = self.nn_ftr_map(x2)
        return rbf_cov(x1, x2, self.length_scale, self.output_scale)




if __name__ == "__main__":
    def forward(x1, x2):
        kernel = JAXRBFKernelNN(input_dim=1,
                                feature_dim=1)
        return kernel(x1, x2)

    rng = jax.random.PRNGKey(42)
    rng, initkey = jax.random.split(rng)
    init, apply = hk.transform(forward)
    x1 = jnp.ones(1)
    x2 = jnp.zeros(1)
    params = init(rng, x1, x2)
    val, grad = jax.value_and_grad(apply)(params, rng, x1, x2)
    print(params)
    print(val)
    print(grad)