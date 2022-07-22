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

    def __init__(
        self,
        mean=None,
        log_stddev=0.0,
        boundary_value=0.0,
        shape=None,
        name="PositiveParameter",
        learnable=True,
    ):
        """
        :param mean: A float specifying the mean of the module
        :param log_stddev: A float specifying the variance in log_scale with which we initialize
            if 0.0, will be deterministically initialized
        :param boundary_value: 
        :param shape: The shape of the parameter. If None, will infer it from the provided (or default) mean 
        :param name: name of the module, if customization is necessary
        :param learnable: whether this is an actual learnable parameter. If not, the module will have the same functionality,
        but it's gradients will all be 0.
        """
        super().__init__(name=name)
        self.learnable = learnable
        if not learnable:
            self.mean = jax.nn.softplus(0.0)  # to be consistent with the logscale defaults
        if mean is None:
            self.log_scale_mean = 0
        else:
            self.log_scale_mean = _softplus_inverse(mean - boundary_value)

        self.log_scale_std = log_stddev
        self.boundary_value = boundary_value
        if shape is not None:
            self.shape = shape
        # otherwise infer shape from mean
        elif isinstance(mean, float) or mean is None:
            self.shape = []
        else:
            self.shape = mean.shape

    def __call__(self):
        if not self.learnable:
            return self.mean

        if self.log_scale_std == 0.0:
            initializer = hk.initializers.Constant(self.log_scale_mean)
        else:
            import numpyro

            initializer = hk.initializers.RandomNormal(self.log_scale_std, self.log_scale_mean)
        exponentiated = jax.nn.softplus(
            hk.get_parameter("__positive_log_scale_param", shape=self.shape, init=initializer)
        )
        return exponentiated + self.boundary_value
