from jax import numpy as jnp
import haiku as hk


class PositiveParameter(hk.Module):
    """A simple module storing a positive parameter. """
    def __init__(self, initial_value, boundary_value=0.0, name="PositiveParameter"):
        super().__init__(name=name)
        self.raw_init = jnp.log(initial_value - boundary_value)
        self.boundary_value = boundary_value
        if isinstance(initial_value, float):
            self.shape = []
        else:
            self.shape = initial_value.shape

    def __call__(self):
        exponentiated = jnp.exp(hk.get_parameter("__positive_log_scale_param", shape=self.shape,
                                                 init=hk.initializers.Constant(self.raw_init)))
        return exponentiated + self.boundary_value
