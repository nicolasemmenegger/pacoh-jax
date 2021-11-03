import functools

import haiku as hk

from pacoh.modules.batching import transform_and_batch_module, multi_transform_and_batch_module
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.modules.exact_gp import JAXExactGP
from pacoh.modules.kernels import JAXRBFKernel
from pacoh.modules.means import JAXConstantMean
from pacoh.models.pure.pure_interfaces import LikelihoodInterface, VanillaGPInterface, VanillaBNNVIInterface
from pacoh.util.constants import MLP_MODULE_NAME


@transform_and_batch_module
def get_pure_batched_nn_functions(output_dim, hidden_layer_sizes, activation):
    def nn_forward(xs):
        nn = hk.nets.MLP(output_sizes=hidden_layer_sizes + (output_dim,), activation=activation)
        return nn(xs)
    return nn_forward


@functools.partial(multi_transform_and_batch_module, num_data_args={'log_prob': 2, 'get_posterior_from_means': 1})
def get_pure_batched_likelihood_functions(likelihood_initial_std, learn_likelihood=True):
    def factory():
        likelihood = JAXGaussianLikelihood(variance=likelihood_initial_std*likelihood_initial_std,
                                           learn_likelihood=learn_likelihood)

        def log_prob(ys_true, ys_pred):
            return likelihood.log_prob(ys_true, ys_pred)

        # add noise to a mean prediction (same as add_noise with a zero_variance_pred_f)
        def get_posterior_from_means(ys_pred):
            return likelihood.get_posterior_from_means(ys_pred)

        return get_posterior_from_means, LikelihoodInterface(log_prob=log_prob,
                                                             get_posterior_from_means=get_posterior_from_means)

    return factory


def construct_vanilla_gp_forward_fns(input_dim, kernel_outputscale, kernel_lengthscale, likelihood_variance):
    def factory():
        # Initialize the mean module with a zero-mean, and use an RBF kernel with *no* learned feature map
        mean_module = JAXConstantMean(0.0)
        covar_module = JAXRBFKernel(input_dim, kernel_lengthscale, kernel_outputscale)
        likelihood = JAXGaussianLikelihood(likelihood_variance)
        gp = JAXExactGP(mean_module,
                        covar_module,
                        likelihood)

        # choose gp.pred_dist as the template function, as it includes the likelihood
        return gp.init_fn, VanillaGPInterface(fit_fn=gp.fit, pred_dist_fn=gp.pred_dist, prior_fn=gp.prior)

    return factory


@functools.partial(multi_transform_and_batch_module, num_data_args={'log_likelihood': 2, 'pred_dist': 1})
def construct_bnn_vi_forward_fns(output_dim, hidden_layer_sizes, activation,
                                 likelihood_initial_std, learn_likelihood=True):
    def factory():
        likelihood = JAXGaussianLikelihood(variance=likelihood_initial_std * likelihood_initial_std,
                                           learn_likelihood=learn_likelihood)

        nn = hk.nets.MLP(name=MLP_MODULE_NAME, output_sizes=hidden_layer_sizes + (output_dim,), activation=activation)

        def pred_dist(xs):
            return likelihood.get_posterior_from_means(nn(xs))

        def log_prob(ys_true, ys_pred):
            return likelihood.log_prob(ys_true, ys_pred)

        return pred_dist, VanillaBNNVIInterface(log_likelihood=log_prob, pred_dist=pred_dist)

    return factory







