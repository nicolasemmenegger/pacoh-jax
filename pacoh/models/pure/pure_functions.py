import functools

import haiku as hk
import jax.nn

from pacoh.modules.batching import (
    multi_transform_and_batch_module,
)
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.modules.exact_gp import JAXExactGP
from pacoh.modules.kernels import JAXRBFKernel, JAXRBFKernelNN, JAXKernel
from pacoh.modules.means import JAXConstantMean, JAXZeroMean, JAXMean
from pacoh.models.pure.pure_interfaces import (
    VanillaGPInterface,
    VanillaBNNVIInterface,
    BaseLearnerInterface,
)
from pacoh.util.constants import MLP_MODULE_NAME


def construct_vanilla_gp_forward_fns(
    input_dim, output_dim,
    kernel_outputscale, kernel_lengthscale, likelihood_variance,
    kernel_log_os_var=0.0, kernel_log_ls_var=0., likelihood_log_var=0.0
):
    def factory():
        # Initialize the mean module with a zero-mean, and use an RBF kernel with *no* learned feature map
        mean_module = JAXZeroMean(output_dim=output_dim)
        covar_module = JAXRBFKernel(input_dim, kernel_lengthscale, kernel_outputscale, kernel_log_ls_var, kernel_log_os_var)
        likelihood = JAXGaussianLikelihood(likelihood_variance, likelihood_log_var)
        gp = JAXExactGP(mean_module, covar_module, likelihood)

        # choose gp.pred_dist as the template function, as it includes the likelihood
        return gp.init_fn, VanillaGPInterface(fit_fn=gp.fit, pred_dist_fn=gp.pred_dist, prior_fn=gp.prior)

    return factory


@functools.partial(
    multi_transform_and_batch_module,
    num_data_args={"log_prob": 2, "pred_dist": 1, "pred": 1},
)
def construct_bnn_forward_fns(
    output_dim,
    hidden_layer_sizes,
    activation,
    likelihood_initial_std,
    learn_likelihood=True,
):
    def factory():
        likelihood_module = JAXGaussianLikelihood(
            variance=likelihood_initial_std * likelihood_initial_std,
            learn_likelihood=learn_likelihood,
        )

        nn = hk.nets.MLP(
            name=MLP_MODULE_NAME,
            output_sizes=hidden_layer_sizes + (output_dim,),
            activation=activation,
        )

        def pred_dist(xs):
            means = nn(xs)
            return likelihood_module(means)  # this adds heteroscedastic variance to all datapoints

        def log_prob(ys_true, ys_pred):
            return likelihood_module.log_prob(ys_true, ys_pred)

        def pred(xs):
            res = nn(xs)
            return res

        return pred_dist, VanillaBNNVIInterface(log_prob=log_prob, pred_dist=pred_dist, pred=pred)

    return factory


def construct_gp_base_learner(
    input_dim,
    output_dim,
    mean_option,
    covar_option,
    learning_mode,
    feature_dim,
    mean_nn_layers,
    kernel_nn_layers,
    learn_likelihood=True,
    initial_noise_std=1.0,
    kernel_length_scale=1.0,
    kernel_output_scale=1.0
):
    def factory():
        """The arguments here are what _setup_gp_prior had."""
        # setup kernel module
        if covar_option == "NN":
            assert learning_mode in [
                "learn_kernel",
                "both",
            ], "neural network parameters must be learned"
            covar_module = JAXRBFKernelNN(input_dim, feature_dim, layer_sizes=kernel_nn_layers, length_scale=kernel_length_scale, output_scale=kernel_output_scale)
        elif covar_option == "SE":
            covar_module = JAXRBFKernel(input_dim, length_scale=kernel_length_scale, output_scale=kernel_output_scale)
        elif callable(covar_option):
            covar_module = covar_option()
            assert isinstance(covar_module, JAXKernel), "Invalid covar_module option"
        else:
            raise ValueError("Invalid covar_module option")

        # setup mean module
        if mean_option == "NN":
            assert learning_mode in [
                "learn_mean",
                "both",
            ], "neural network parameters must be learned"
            mean_module = hk.nets.MLP(output_sizes=mean_nn_layers + (output_dim,), activation=jax.nn.tanh)
        elif mean_option == "constant":
            mean_module = JAXConstantMean(output_dim=output_dim)
        elif mean_option == "zero":
            mean_module = JAXZeroMean(output_dim=output_dim)
        elif callable(mean_option):
            assert isinstance(mean_option, JAXMean), "Invalid mean_module option"
            mean_module = mean_option
        else:
            raise ValueError("Invalid mean_module option")

        likelihood = JAXGaussianLikelihood(
            output_dim=output_dim,
            variance=initial_noise_std * initial_noise_std,
            learn_likelihood=learn_likelihood,
            variance_constraint_gt=1e-3,
        )

        base_learner = JAXExactGP(mean_module, covar_module, likelihood)

        init_fn = base_learner.init_fn
        base_learner_fit = base_learner.fit
        base_learner_predict = base_learner.pred_dist

        def base_learner_mll_estimator(xs, ys):
            return base_learner.marginal_ll(xs, ys)

        return init_fn, BaseLearnerInterface(
            base_learner_fit=base_learner_fit,
            base_learner_predict=base_learner_predict,
            base_learner_mll_estimator=base_learner_mll_estimator,
        )

    return factory
