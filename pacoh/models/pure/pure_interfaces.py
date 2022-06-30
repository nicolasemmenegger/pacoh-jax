from typing import NamedTuple, Callable


class NNBaseLearner(NamedTuple):
    """
    This is a neural network base learner. Note that it doesn't have a fit method when compared to the GPBaseLearner
    This is because we fit by doing gradient-based optimization outside of the impure hk.Modules, and hence only need access to a forward function
    pred_mean and pred_dist for BNNs.
    """

    pred_dist: Callable
    pred_mean: Callable
    log_prob: Callable


class GPBaseLearner(NamedTuple):
    """
    This is a base learner with (potentially) learnable kernel, mean and likelihood modules, and
    this class provides the interfaces that are transformed by hk.tansform and similar methods

    this is the interface PACOH modules learners should provide.
    hyper_prior_ll: A function that yields the log likelihood of the prior parameters under the hyperprior
    base_learner_fit: Fits the modules learner to some data # maybe I need state here
    base_learner_predict: Actual predict on a task
    base_learner_mll_estimator: The mll of the modules estimator under the data one just passed it
    """

    base_learner_fit: Callable
    base_learner_predict: Callable
    base_learner_mll_estimator: Callable
