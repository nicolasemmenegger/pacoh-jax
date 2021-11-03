from typing import TypeVar, Callable, Union, Tuple

import jax.numpy as jnp
from numpyro.distributions import Distribution

# from pacoh.models.regression_base import RegressionModel


Tree = TypeVar('Tree')
NormalizedPredFunc = Callable[[object, jnp.array, bool, ...],
                              Union[Distribution, Tuple[jnp.array, jnp.array]]]
RawPredFunc = Callable[[object, jnp.array, ...], Distribution]
