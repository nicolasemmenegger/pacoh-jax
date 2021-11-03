from typing import NamedTuple, Any, Optional, Callable

import jax
from numpyro.distributions import Distribution
from jax import numpy as jnp

from pacoh.util.typing import Tree


class VanillaGPInterface(NamedTuple): # TODO: This can probably be called statefulbaselearnerinterface or something like this
    fit_fn: Any # fit
    pred_dist_fn: Any
    prior_fn: Any

class VanillaBNNVIInterface(NamedTuple): # TODO: This can probably be called BaseLearnerInterface
    pred_dist: Any # Callable[[Tree, jnp.array], Distribution]
    pred: Any # Callable[[Tree, jnp.array], Distribution]
    log_prob: Any

class LikelihoodInterface(NamedTuple):
    log_prob: Any
    get_posterior_from_means: Any