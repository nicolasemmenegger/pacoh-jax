from typing import NamedTuple, Any


class VanillaGPInterface(NamedTuple): # TODO: This can probably be called statefulbaselearnerinterface or something like this
    fit_fn: Any # fit
    pred_dist_fn: Any
    prior_fn: Any

class VanillaBNNVIInterface(NamedTuple): # TODO: This can probably be called BaseLearnerInterface
    pred_dist: Any
    log_likelihood: Any

class LikelihoodInterface(NamedTuple):
    log_prob: Any
    get_posterior_from_means: Any