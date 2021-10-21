import functools

from typing import NamedTuple, Any
import haiku as hk
from pacoh.modules.batched_modules import transform_and_batch_module, multi_transform_and_batch_module
from pacoh.modules.distributions import JAXGaussianLikelihood


@transform_and_batch_module
def get_pure_batched_nn_functions(output_dim, hidden_layer_sizes, activation):
    def nn_forward(xs):
        nn = hk.nets.MLP(output_sizes=hidden_layer_sizes + (output_dim,), activation=activation)
        return nn(xs)
    return nn_forward


class LikelihoodInterface(NamedTuple):
    log_prob: Any
    get_posterior_from_means: Any


@functools.partial(multi_transform_and_batch_module, num_data_args={'log_prob': 2, 'get_posterior_from_means': 1})
def get_pure_batched_likelihood_functions(likelihood_initial_std):
    def factory():
        likelihood = JAXGaussianLikelihood(variance=likelihood_initial_std*likelihood_initial_std)

        def log_prob(ys_true, ys_pred):
            return likelihood.log_prob(ys_true, ys_pred)

        def get_posterior_from_means(ys_pred):  # add noise to a mean prediction (same as add_noise with a zero_variance_pred_f)
            return likelihood.get_posterior_from_means(ys_pred)

        return get_posterior_from_means, LikelihoodInterface(log_prob=log_prob,
                                                             get_posterior_from_means=get_posterior_from_means)

    return factory
