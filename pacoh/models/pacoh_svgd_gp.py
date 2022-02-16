import functools
from typing import Callable, Collection, Union

import jax.random
import numpy as np
from jax import numpy as jnp

from pacoh.algorithms.svgd import SVGD
from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.modules.belief import GaussianBelief, GaussianBeliefState
from pacoh.util.constants import (
    LIKELIHOOD_MODULE_NAME,
    MLP_MODULE_NAME,
    POSITIVE_PARAMETER_NAME,
    KERNEL_MODULE_NAME,
    MEAN_MODULE_NAME,
)
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.util.distributions import get_mixture, vmap_dist
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel, get_pytree_rbf_fn
from pacoh.util.initialization import (
    initialize_batched_model_with_state,
    initialize_optimizer,
)

from pacoh.models.pure.pure_functions import construct_gp_base_learner
from pacoh.modules.batching import multi_transform_and_batch_module_with_state
from pacoh.util.tree import pytree_unstack

construct_meta_gp_forward_fns = multi_transform_and_batch_module_with_state(
    construct_gp_base_learner,
    num_data_args={
        "base_learner_fit": 2,
        "base_learner_predict": 1,
        "base_learner_mll_estimator": 2,
    },
)

# def foo(mod_name: str, name: str, value):
#     if mod_name == "jax_gaussian_likelihood/~/PositiveParameter":
#         return value * 0.0 + -2.0
#     elif mod_name == "jaxrbf_kernel/~/LengthScale":
#         return value * 0.0 - 1.1
#     elif mod_name == "jaxrbf_kernel/~/OutputScale":
#         return value * 0.0 + 0.5
#     else:
#         assert False, "boum"


class PACOH_SVGD_GP(RegressionModelMetaLearned):
    def __init__(
        self,
        # sizing options
        input_dim: int,
        output_dim: int,
        feature_dim: int = 2,
        num_tasks: int = None,
        # hyperprior structure specification
        covar_module: Union[str, Callable[[], JAXKernel]] = "NN",
        mean_module: Union[str, Callable[[], JAXMean]] = "NN",
        learning_mode: str = "both",
        mean_nn_layers: Collection[int] = (32, 32),
        kernel_nn_layers: Collection[int] = (32, 32),
        # hyperprior detailed specification
        weight_prior_std=0.5,
        bias_prior_std: float = 3.0,
        likelihood_prior_mean: float = 0.01,
        likelihood_prior_std: float = 0.1,
        kernel_prior_mean: float = 1.0,
        kernel_prior_std: float = 1.0,
        mean_module_prior_mean: float = 1.0,
        mean_module_prior_std: float = 1.0,
        learn_likelihood: bool = True,
        # inference algorithm specification
        svgd_kernel: str = "RBF",
        svgd_kernel_bandwidth=100.0,
        optimizer="AdamW",
        lr: float = 1e-3,
        lr_decay: float = 1.0,
        weight_decay: float = 0.0,
        prior_weight: float = 1e-4,
        num_particles=10,
        # train loop specification
        num_iter_meta_fit: int = 10000,
        task_batch_size: int = -1,
        minibatch_at_dataset_level: bool = True,
        dataset_batch_size: int = -1,
        # data handling options
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        random_state: jax.random.PRNGKey = None,
    ):
        """
        The std parameters of the likelihood and kernel module are actually in logscale (anything that represents a
        positive parameter)
        """
        super().__init__(
            input_dim,
            output_dim,
            normalize_data,
            normalizer,
            random_state,
            task_batch_size,
            num_tasks,
            flatten_ys=True,
            minibatch_at_dataset_level=minibatch_at_dataset_level,
            dataset_batch_size=dataset_batch_size,
        )

        # 0) check options
        assert mean_module in ["NN", "constant", "zero"] or isinstance(
            mean_module, JAXMean
        ), "Invalid mean_module option"
        assert covar_module in ["NN", "SE"] or isinstance(
            covar_module, JAXKernel
        ), "Invalid covar_module option"
        assert optimizer in ["AdamW", "Adam", "SGD"], "Invalid optimizer option"
        assert learning_mode in ["mean", "kernel", "both"], "Invalid learning mode"

        # a) useful attributes
        self._num_particles = num_particles

        # b) get batched forward functions for nparticle models in parallel
        init, self._apply, self._apply_broadcast = construct_meta_gp_forward_fns(
            input_dim,
            output_dim,
            mean_module,
            covar_module,
            learning_mode,
            feature_dim,
            mean_nn_layers,
            kernel_nn_layers,
            learn_likelihood,
            initial_noise_std=likelihood_prior_mean,
        )

        # c) initialize the the state of the hyperprior and of the (hyper posterior) particles
        self._rng, init_key = jax.random.split(self._rng)
        params, template, self._empty_states = initialize_batched_model_with_state(
            init, num_particles, init_key, (self._task_batch_size, input_dim)
        )
        self._single_empty_state = pytree_unstack(self._empty_states)

        def mean_std_map(mod_name: str, name: str, _: jnp.array):
            # for each module and parameter, we can specify custom mean and std for the hyperprior
            transform = lambda value, name: np.log(value) if POSITIVE_PARAMETER_NAME in name else value
            # transform positive parameters to log_scale for storage
            if LIKELIHOOD_MODULE_NAME in mod_name:
                return transform(likelihood_prior_mean, name), likelihood_prior_std
            elif MLP_MODULE_NAME in mod_name:
                if name == "w":
                    return 0.0, weight_prior_std
                elif name == "b":
                    return 0.0, bias_prior_std
                else:
                    raise AssertionError("Unknown MLP parameter name")
            elif KERNEL_MODULE_NAME in mod_name:
                # lengthscale and outputscale
                return transform(kernel_prior_mean, name), kernel_prior_std
            elif MEAN_MODULE_NAME in mod_name:
                return mean_module_prior_mean, mean_module_prior_std
            else:
                raise AssertionError("Unknown hk.Module: can only handle mlp and likelihood")

        self.hyperprior = GaussianBeliefState.initialize_heterogenous(mean_std_map, template)
        self._rng, particle_sample_key = jax.random.split(self._rng)

        self.particles = GaussianBelief.rsample(self.hyperprior, particle_sample_key, num_particles)
        # self.particles = hk.data_structures.map(foo, self.particles)

        # d) setup kernel forward function and the base learner partition function
        if svgd_kernel != "RBF":  # elif kernel == 'IMQ':
            raise NotImplementedError("IMQ and other options not yet supported")

        if not minibatch_at_dataset_level:
            raise NotImplementedError()

        # e) setup svgd target
        def target_post_prob_batched(hyper_posterior_particles, rngs, *data, mll_many_many):
            # mll_many_many is expected to produce a matrix (i,j) |-> ln(Z(S_j,P_i)) where
            meta_xs, meta_ys = data
            # this will have size K x task_batch
            mll_matrix = mll_many_many(hyper_posterior_particles, self._empty_states, None, meta_xs, meta_ys)[
                0
            ]
            batch_data_likelihood_per_particle = jnp.sum(mll_matrix, axis=0) * num_tasks / task_batch_size
            hyperprior_log_prob = GaussianBelief.log_prob(self.hyperprior, hyper_posterior_particles)
            return batch_data_likelihood_per_particle + prior_weight * hyperprior_log_prob

        # f) setup optimizer
        self.optimizer, self.optimizer_state = initialize_optimizer(
            optimizer, lr, self.particles, lr_decay=lr_decay, weight_decay=weight_decay
        )

        # g) thread together svgd
        mll_many_many = jax.jit(
            vmap_dist(self._apply.base_learner_mll_estimator, (None, None, None, 0, 0), 0)
        )
        self.svgd = SVGD(
            functools.partial(target_post_prob_batched, mll_many_many=mll_many_many),
            jax.jit(get_pytree_rbf_fn(svgd_kernel_bandwidth, 1.0)),
            self.optimizer,
            self.optimizer_state,
        )

    def meta_fit(
        self,
        meta_train_tuples,
        meta_valid_tuples=None,
        log_period=500,
        num_iter_fit=None,
    ):
        super().meta_fit(
            meta_train_tuples,
            meta_valid_tuples,
            log_period=log_period,
            num_iter_fit=num_iter_fit,
        )

    def _meta_step(self, minibatch):
        xs_tasks, ys_tasks = minibatch
        neg_log_prob, self.particles = self.svgd.step(self.particles, xs_tasks, ys_tasks)
        return neg_log_prob

    def _recompute_posterior(self):
        # use the stored data in xs_data, ys_data to instantiate a base_learner
        _, self._states = self._apply.base_learner_fit(
            self.particles, self._empty_states, None, self._xs_data, self._ys_data
        )

    @normalize_predict
    def predict(self, xs):
        pred_dist, _ = self._apply.base_learner_predict(self.particles, self._states, None, xs)
        return get_mixture(pred_dist, self._num_particles)


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_debug_nans", True)
    config.update("jax_disable_jit", False)

    from experiments.data_sim import SinusoidDataset

    data_sim = SinusoidDataset(random_state=np.random.RandomState(29))
    n_tasks = 20
    meta_train_data = data_sim.generate_meta_train_data(n_tasks=n_tasks, n_samples=10)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=50, n_samples_context=10, n_samples_test=50)

    NN_LAYERS = (32, 32, 32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title("sample from the GP prior")
        plt.show()
    """ 2) Classical mean learning based on mll """

    print("\n ---- GPR mll meta-learning ---- ")

    for weight_decay in [0.5]:
        pacoh_svgd = PACOH_SVGD_GP(
            1,
            1,
            num_tasks=n_tasks,
            learning_mode="both",
            weight_decay=weight_decay,
            task_batch_size=5,
            covar_module="NN",
            mean_module="constant",
            mean_nn_layers=NN_LAYERS,
            feature_dim=2,
            svgd_kernel_bandwidth=1000.0,
            kernel_nn_layers=NN_LAYERS,
            num_particles=20,
            learn_likelihood=True,
        )

        itrs = 0
        print("---- weight-decay =  %.4f ----" % weight_decay)

        for i in range(40):
            n_iter = 500
            pacoh_svgd.meta_fit(
                meta_train_data,
                meta_valid_tuples=meta_test_data,
                log_period=100,
                num_iter_fit=n_iter,
            )
            itrs += n_iter

            x_plot = np.linspace(-5, 5, num=150)
            x_context, y_context, x_test, y_test = meta_test_data[0]
            # do a target fit
            pacoh_svgd._clear_data()
            pacoh_svgd.add_data_points(x_context, y_context, refit=True)
            # do a target predict
            pred_mean, pred_std = pacoh_svgd.predict(x_plot, return_density=False)
            plt.scatter(x_test, y_test, color="green")  # the unknown target test points
            plt.scatter(x_context, y_context, color="red")  # the target train points
            plt.plot(x_plot, pred_mean)  # the curve we fitted based on the target test points
            lcb, ucb = pacoh_svgd.confidence_intervals(x_plot)
            plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.2, color="green")
            plt.title("GPR meta mll (weight-decay =  %.4f) itrs = %i" % (weight_decay, itrs))
            plt.show()
