import functools
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional

import haiku as hk
import jax

from meta_bo.models.base.common import PositiveParameter
from meta_bo.models.base.neural_network import JAXNeuralNetwork
from jax import numpy as jnp


def rbf_cov(x1, x2, ls_param, os_param):
    return os_param() * jnp.exp(-0.5 * (jnp.linalg.norm(x1 - x2) ** 2) / (ls_param() ** 2))


class JAXKernel(hk.Module):
    """ Base class for kernels that supports composition with a learned feature map
        Note:
            Concrete implementation should subclass kernel_fn
    """
    def __init__(self,
                 input_dim,
                 covar_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.float64],
                 feature_dim: Optional[Callable] = None,
                 feature_map: Optional[Callable] = None):
        super().__init__()  # haiku module initializer
        assert (feature_map is not None and feature_dim is not None) or (feature_map is None and feature_dim is None), \
            "either both the map and feature dim have to be none, or neither should"

        warnings.warn("I should probably switch feature dim and input dim")
        # kernel module
        self._covar_fn = covar_fn
        self._input_dim = input_dim

        self._feature_map = feature_map if feature_map is not None else (lambda x: x)
        self._feature_dim = input_dim if feature_dim is None else feature_dim

    def __call__(self, x1: jnp.ndarray, x2: Optional[jnp.ndarray] = None):
        return self._covar_fn(self._feature_map(x1), self._feature_map(x2))


class JAXRBFKernel(JAXKernel):
    def __init__(self,
                 input_dim,
                 length_scale=1.0,
                 output_scale=1.0,
                 length_scale_constraint_gt=0.0,
                 output_scale_constraint_gt=0.0,
                 feature_dim: Optional[int] = None,
                 feature_map: Optional[Callable] = None):
        # Temporary hack => call with no setup covariance function
        super().__init__(input_dim, None)

        self.output_scale = PositiveParameter(initial_value=output_scale,
                                              boundary_value=output_scale_constraint_gt,
                                              name="OutputScale")
        self.length_scale = PositiveParameter(initial_value=length_scale,
                                              boundary_value=length_scale_constraint_gt,
                                              name="LengthScale")
        covar_fn = functools.partial(rbf_cov, ls_param=self.length_scale, os_param=self.output_scale)
        self._covar_fn = covar_fn
        self._feature_dim = feature_dim


class JAXRBFKernelNN(JAXRBFKernel):
    def __init__(self,
                 input_dim,
                 feature_dim,
                 layer_sizes=(64, 64),
                 length_scale=1.0,
                 output_scale=1.0,
                 length_scale_constraint_gt=0.0,
                 output_scale_constraint_gt=0.0):
        super().__init__(input_dim,
                         length_scale,
                         output_scale,
                         length_scale_constraint_gt,
                         output_scale_constraint_gt,
                         feature_dim=feature_dim,
                         feature_map=None)
        self.nn = hk.nets.MLP(output_sizes=layer_sizes + (feature_dim,), activation=jax.nn.tanh)
        self._feature_map = self.nn

def forward(x1, x2):
    kernel = JAXRBFKernelNN(input_dim=1,
                             feature_dim=1)
    return kernel(x1, x2)

if __name__ == "__main__":
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
