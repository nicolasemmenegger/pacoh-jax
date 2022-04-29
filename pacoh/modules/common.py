import jax
from jax import numpy as jnp
import haiku as hk


def _softplus_inverse(x):
    return jnp.log(jnp.exp(x) - 1.0)


class PositiveParameter(hk.Module):
    """
    A simple module storing a positive parameter. Can specify a mean and variance. The raw parameter is stored
    in log scale, more specifically, param = softplus(raw_parameter) + boundary_constraint, which is consistent
    with gpytorch.
    """

    def __init__(self, mean=None, log_variance=0.0, boundary_value=0.0, shape=None, name="PositiveParameter"):
        """
        :param mean: A float specifying the mean of the module
        :param log_variance: A float specifying the variance in log_scale with which we initialize
            if 0.0, will be deterministically initialized
        :param boundary_value:
        :param shape
        :param name: name of the module
        """
        super().__init__(name=name)
        if mean is None:
            self.log_scale_mean = 0
        else:
            self.log_scale_mean = _softplus_inverse(mean - boundary_value)
        self.log_scale_var = log_variance
        self.boundary_value = boundary_value
        if shape is not None:
            self.shape = shape
        # otherwise infer shape from mean
        elif isinstance(mean, float) or mean is None:
            self.shape = []
        else:
            self.shape = mean.shape

    def __call__(self):
        if self.log_scale_var == 0.0:
            initializer = hk.initializers.Constant(self.log_scale_mean)
        else:
            initializer = hk.initializers.RandomNormal(self.log_scale_mean, self.log_scale_var)
        exponentiated = jax.nn.softplus(
            hk.get_parameter("__positive_log_scale_param", shape=self.shape, init=initializer)
        )
        return exponentiated + self.boundary_value
