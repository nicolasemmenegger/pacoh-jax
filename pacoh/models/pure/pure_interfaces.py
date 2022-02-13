from typing import NamedTuple, Any, Callable, Tuple

from jax import numpy as jnp


class VanillaGPInterface(NamedTuple):
    fit_fn: Any  # fit
    pred_dist_fn: Any
    prior_fn: Any


class VanillaBNNVIInterface(NamedTuple):
    pred_dist: Any  # Callable[[Tree, jnp.array], Distribution]
    pred: Any  # Callable[[Tree, jnp.array], Distribution]
    log_prob: Any


class LikelihoodInterface(NamedTuple):
    log_prob: Any
    get_posterior_from_means: Any


class BaseLearnerInterface(NamedTuple):
    """
    kernel, mean and likelihood, but needs to perform target inference in parallel."""

    """This is the interface PACOH modules learners should provide.
    hyper_prior_ll: A function that yields the log likelihood of the prior parameters under the hyperprior
    base_learner_fit: Fits the modules learner to some data # maybe I need state here
    base_learner_predict: Actual predict on a task
    base_learner_mll_estimator: The mll of the modules estimator under the data one just passed it
    """
    base_learner_fit: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], None]
    base_learner_predict: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]
    base_learner_mll_estimator: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], jnp.float32]
