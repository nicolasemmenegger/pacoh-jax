import warnings
from typing import Callable, NamedTuple, Tuple

import haiku as hk
from jax import numpy as jnp

from meta_bo.models.base.distributions import JAXGaussianLikelihood
from meta_bo.models.base.gp_components import JAXConstantMean, JAXZeroMean, JAXMean, JAXExactGP
from meta_bo.models.base.kernels import JAXRBFKernelNN, JAXRBFKernel, JAXKernel
from meta_bo.models.base.neural_network import JAXNeuralNetwork

class PACOHInterface(NamedTuple):
    """TODO this needs to be vectorized in some way. The MAP version needs to share the same cholesky accross calls
    kernel, mean and likelihood, but needs to perform target inference in parallel. """
    """This is the interface PACOH base learners should provide.
    hyper_prior_ll: A function that yields the log likelihood of the prior parameters under the hyperprior
    base_learner_fit: Fits the base learner to some data # maybe I need state here
    base_learner_predict: Actual predict on a task
    base_learner_mll_estimator: The mll of the base estimator under the data one just passed it
    """
    #base_learner_fit: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], None]
    #base_learner_predict: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], None]
    base_learner_fit_and_predict: Callable[[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], None]
    #hyper_prior_log_ll: Callable[[], jnp.float32]
    base_learner_mll_estimator: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], jnp.float32]


def construct_pacoh_map_forward_fns(input_dim, mean_option, covar_option, learning_mode,
                                    feature_dim, mean_nn_layers, kernel_nn_layers):
    def factory():
        """The arguments here are what _setup_gp_prior had. Maybe they need to be factories tho"""
        # setup kernel module
        if covar_option == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            covar_module = JAXRBFKernelNN(input_dim, feature_dim, layer_sizes=kernel_nn_layers)
        elif covar_option == 'SE':
            covar_module = JAXRBFKernel(input_dim=input_dim)
        elif callable(covar_option):
            covar_module = covar_option()
            assert isinstance(covar_module, JAXKernel), "Invalid covar_module option"
        else:
            raise ValueError('Invalid covar_module option')

        # setup mean module
        if mean_option == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            mean_module = JAXNeuralNetwork(input_dim=input_dim, output_dim=1, layer_sizes=mean_nn_layers)
        elif mean_option == 'constant':
            mean_module = JAXConstantMean()
        elif mean_option == 'zero':
            mean_module = JAXZeroMean()
        elif callable(mean_option):
            assert isinstance(mean_option, JAXMean), "Invalid mean_module option"
            mean_module = mean_option
        else:
            raise ValueError('Invalid mean_module option')

        likelihood = JAXGaussianLikelihood(variance_constraint_gt=1e-3)
        base_learner = LearnedGPRegressionModel(mean_module, covar_module, likelihood)

        def init_fn(task):
            """This is somewhat uninstructive """
            base_learner.fit(*task)

        def base_learner_fit_and_predict(context_task, test_x):
            base_learner.fit(*context_task)
            return base_learner.pred_dist(test_x)

        def base_learner_mll_estimator(task):
            # is this the same state
            base_learner.fit(*task)
            return base_learner.marginal_ll()

        return init_fn, PACOHInterface(base_learner_fit_and_predict=base_learner_fit_and_predict,
                                       base_learner_mll_estimator=base_learner_mll_estimator)

    return factory
