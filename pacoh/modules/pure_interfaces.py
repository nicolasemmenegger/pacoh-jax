from typing import NamedTuple, Any


class VanillaGPInterface(NamedTuple):
    fit_fn: Any
    pred_dist_fn: Any
    prior_fn: Any

class VanillaBNNVIInterface(NamedTuple):
    pred_dist: Any
    log_likelihood: Any

class LikelihoodInterface(NamedTuple):
    log_prob: Any
    get_posterior_from_means: Any