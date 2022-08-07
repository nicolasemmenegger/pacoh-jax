from abc import abstractmethod

import haiku as hk
from jax import numpy as jnp


class JAXMean(hk.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, x):
        pass


class JAXConstantMean(JAXMean):
    """Mean module with a learnable mean."""

    def __init__(self, output_dim=1, initial_constant=0.0, learnable=True, initialization_std=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.learnable = learnable
        self.initialization_std = initialization_std
        if isinstance(initial_constant, float):
            self.init_constant = jnp.ones((output_dim,)) * initial_constant
        else:
            self.init_constant = initial_constant

    def __call__(self, xs):
        # works for both batch or unbatched
        if not self.learnable:
            return self.init_constant
        if self.initialization_std == 0.0:
            mean = hk.get_parameter(
                "mu",
                shape=[self.output_dim],
                init=hk.initializers.Constant(self.init_constant),
                dtype=jnp.float64
            )
        else:
            mean = hk.get_parameter(
                "mu",
                shape=[self.output_dim],
                init=hk.initializers.RandomNormal(self.initialization_std, self.init_constant),
                dtype=jnp.float64
            )
        return jnp.ones(xs.shape[:-1] + (self.output_dim,)) * mean


class JAXZeroMean(JAXMean):
    """Always zero, not learnable."""

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def __call__(self, xs):
        return jnp.ones(xs.shape[:-1] + (self.output_dim,)) * 0.0
