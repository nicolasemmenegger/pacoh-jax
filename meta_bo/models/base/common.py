from jax import numpy as jnp
import haiku as hk


class PositiveParameter(hk.Module):
    """A simple module storing a positive parameter. """
    def __init__(self, initial_value, dtype=jnp.float64, shape=[], boundary_value=0.0):
        self.raw_init = jnp.log(initial_value - boundary_value)
        self.boundary_value = boundary_value
        self.shape = shape
        self.dtype = dtype

    def __call__(self):
        raw = jnp.exp(hk.get_parameter("_raw_param", shape=self.shape, dtype=self.dtype, init=self.raw_init))
        return jnp.exp(raw) + self.boundary_value
