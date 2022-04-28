from typing import NamedTuple, Any, Callable, Tuple

from jax import numpy as jnp
from numpyro.distributions import Distribution


class NNBaseLearner(NamedTuple):
    pred_dist: Any
    pred: Any
    log_prob: Any


class GPBaseLearner(NamedTuple):
    """
    kernel, mean and likelihood, but needs to perform target inference in parallel."""

    """This is the interface PACOH modules learners should provide.
    hyper_prior_ll: A function that yields the log likelihood of the prior parameters under the hyperprior
    base_learner_fit: Fits the modules learner to some data # maybe I need state here
    base_learner_predict: Actual predict on a task
    base_learner_mll_estimator: The mll of the modules estimator under the data one just passed it
    """
    base_learner_fit: Callable[[jnp.ndarray, jnp.ndarray], None]
    base_learner_predict: Callable[[jnp.ndarray], Distribution]
    base_learner_mll_estimator: Callable[[jnp.ndarray, jnp.ndarray], jnp.float32]
