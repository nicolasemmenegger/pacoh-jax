import functools
import time
import warnings
from typing import Callable, Collection, Union

import jax.random
import numpy as np
import optax
import torch
from absl import logging
from jax import numpy as jnp

from pacoh.algorithms.svgd import SVGD
from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.modules.belief import GaussianBelief, GaussianBeliefState
from pacoh.util.constants import LIKELIHOOD_MODULE_NAME, MLP_MODULE_NAME, POSITIVE_PARAMETER_NAME, KERNEL_MODULE_NAME, \
    MEAN_MODULE_NAME
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.modules.distributions import get_mixture
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel, pytree_rbf_set
from pacoh.util.initialization import initialize_batched_model, initialize_batched_model_with_state

from pacoh.models.pure.pure_functions import construct_pacoh_map_forward_fns
from pacoh.modules.batching import multi_transform_and_batch_module_with_state

# this is all there is to it
construct_meta_gp_forward_fns = multi_transform_and_batch_module_with_state(construct_pacoh_map_forward_fns)

class PACOH_SVGD_GP(RegressionModelMetaLearned):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 weight_prior_std=0.5,
                 bias_prior_std=3.0,
                 likelihood_prior_mean: float = 0.1,
                 likelihood_prior_std: float = 1.0,
                 kernel_prior_mean: float = 1.0,
                 kernel_prior_std: float = 1.0,
                 mean_module_prior_mean: float = 1.0,
                 mean_module_prior_std: float = 1.0,
                 svgd_kernel: str = 'RBF',
                 svgd_kernel_bandwidth=100.,
                 num_particles = 10,
                 optimizer='Adam',
                 weight_decay: float = 0.0,
                 prior_weight: float = 1e-3,
                 feature_dim: int = 2,
                 num_iter_fit: int = 10000,
                 learn_likelihood: bool = True,
                 covar_module: Union[str, Callable[[], JAXKernel]] = 'NN',  # TODO fix type annotation
                 mean_module: Union[str, Callable[[], JAXMean]] = 'NN',
                 learning_mode: str = 'both',
                 mean_nn_layers: Collection[int] = (32, 32),
                 kernel_nn_layers: Collection[int] = (32, 32),
                 task_batch_size: int = -1,
                 num_tasks: int = 1,
                 vectorize_over_tasks=True,
                 lr: float = 1e-3,
                 lr_decay: float = 1.0,
                 normalize_data: bool = True,
                 normalizer: DataNormalizer = None,
                 random_state: jax.random.PRNGKey = None):
        """
        Notes: This is an implementation that does minibatching both at the task and at the dataset level


        """
        super().__init__(input_dim, output_dim, normalize_data, normalizer, random_state)

        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, JAXMean), 'Invalid mean_module option'
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, JAXKernel), 'Invalid covar_module option'
        assert optimizer in ['Adam', 'SGD']

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_dim = feature_dim
        self.num_iter_fit = num_iter_fit
        self.task_batch_size = task_batch_size
        self.num_tasks = num_tasks
        self.mean_nn_layers = mean_nn_layers
        self.kernel_nn_layers = kernel_nn_layers

        # a) meta learning setup
        self.num_iter_fit, self.prior_weight, self.feature_dim = num_iter_fit, prior_weight, feature_dim
        self.weight_prior_std, self.bias_prior_std = weight_prior_std, bias_prior_std
        self.num_particles = num_particles
        if task_batch_size < 1:
            self.task_batch_size = len(meta_train_data)
        else:
            self.task_batch_size = min(task_batch_size, len(meta_train_data))

        # b) get batched forward functions for nparticle models in parallel
        init, self.apply, self.apply_broadcast = construct_meta_gp_forward_fns(
            input_dim, output_dim, mean_module, covar_module, learning_mode, feature_dim,
            mean_nn_layers, kernel_nn_layers, learn_likelihood)

        # c) initialize the the state of the hyperprior and of the hyperposterior particles
        self._rng, init_key = jax.random.split(self._rng)
        params, template, states = initialize_batched_model_with_state(init, num_particles, init_key, (self.task_batch_size, input_dim))

        def mean_std_map(mod_name: str, name: str, _: jnp.array):
            """ This function specifies the hyperposterior for each of the components of the model.  """
            # TODO maybe like a dictionary would be easier to pass things around
            transform = lambda value, name: np.log(value) if POSITIVE_PARAMETER_NAME in name else value
            # transform positive parameters to log_scale for storage

            if LIKELIHOOD_MODULE_NAME in mod_name:
                return likelihood_prior_mean, likelihood_prior_std
            elif MLP_MODULE_NAME in mod_name:
                if name == "w":
                    return 0.0, weight_prior_std
                elif name == "b":
                    return 0.0, bias_prior_std
                else:
                    raise AssertionError("Unknown MLP parameter name")
            elif KERNEL_MODULE_NAME in mod_name:
                # lengthscale and ouputscale
                return transform(kernel_prior_mean, name), kernel_prior_std
            elif MEAN_MODULE_NAME in mod_name:
                return mean_module_prior_mean, mean_module_prior_std
            else:
                raise AssertionError("Unknown hk.Module: can only handle mlp and likelihood")

        self.hyperprior = GaussianBeliefState.initialize_heterogenous(mean_std_map, template)
        self._rng, particle_sample_key = jax.random.split(self._rng)
        self.particles = GaussianBelief.rsample(self.hyperprior, particle_sample_key, num_particles)

        # d) setup kernel forward function and the base learner log likelihood function
        if svgd_kernel != "RBF":  # elif kernel == 'IMQ':
            raise NotImplementedError("IMQ and other options not yet supported")

        def kernel_fwd(particles):
            return pytree_rbf_set(particles, particles, length_scale=svgd_kernel_bandwidth,
                                  output_scale=1.0)

        # e) setup all the forward functions needed by the SVGD class.
        def target_post_prob_batched(hyper_posterior_particles, rngs, *data, mll_many_many):
            meta_xs, meta_ys = data # meta_xs/meta_ys should have size (task_batch_size, batch_size, input_dim/output_dim)
            mll_matrix = mll_many_many(hyper_posterior_particles, None, meta_xs, meta_ys) # this will have size Kxtask_batch
            batch_data_likelihood_per_particle = jnp.sum(mll_matrix, axis=0) * num_tasks/task_batch_size  # TODO check axis
            hyperprior_log_prob = GaussianBelief.log_prob(self.hyperprior, hyper_posterior_particles)
            return batch_data_likelihood_per_particle + prior_weight + hyperprior_log_prob

        """ Optimizer setup """
        if lr_decay < 1.0:
            self.lr_scheduler = optax.exponential_decay(lr, 1000, decay_rate=lr_decay, staircase=True)
        else:
            self.lr_scheduler = optax.constant_schedule(lr)

        assert optimizer in ['Adam', 'SGD'], "Optimizer option not supported"
        if optimizer == "SGD":
            self.optimizer = optax.sgd(self.lr_scheduler)
        else:
            self.optimizer = optax.adam(self.lr_scheduler)

        self.optimizer_state = self.optimizer.init(self.particles)

        # self.apply.base_learner_mll is already batched along the parameter dimensions. Now we batch it along xs, ys
        if not vectorize_over_tasks:
            raise NotImplementedError()

        mll_many_many = jax.jit(jax.vmap(self.apply.base_learner_mll_estimator, (None, None, 0, 0), 1))
        self.svgd = SVGD(functools.partial(target_post_prob_batched, mll_many_many=mll_many_many),
                         jax.jit(kernel_fwd), self.optimizer, self.optimizer_state)

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, log_period=500, num_iter_fit=None):
        super().meta_fit(meta_train_tuples, meta_valid_tuples, log_period=log_period, num_iter_fit=n_iter)

    def _meta_step(self, xs_tasks, ys_tasks):
        self.svgd.step(self.particles, xs_tasks, ys_tasks)

    def _recompute_posterior(self):
        # use the stored data in xs_data, ys_data to instantiate a base_learner
        self._rng, fitkey = self._rng.split()
        self.apply.base_learner_fit(self.particles,
                                    self.state,
                                    fitkey,
                                    self.xs_data,
                                    self.ys_data)

    @normalize_predict
    def predict(self, xs, return_density=False):
        return get_mixture(self.apply.pred(self.particles, self.state, None, xs), self.num_particles)

    #
    # def _prepare_meta_train_tasks(self, meta_train_tuples, flatten_y=True):
    #     self._check_meta_data_shapes(meta_train_tuples)
    #     if self._normalization_stats is None:
    #         self._compute_meta_normalization_stats(meta_train_tuples)
    #     else:
    #         self._set_normalization_stats(self._normalization_stats)
    #
    #     if not self.fitted:
    #         self._rng, init_rng = jax.random.split(self._rng) # random numbers
    #         self.params, self.empty_state = self._init_fn(init_rng, meta_train_tuples[0][0]) # prior parameters, initial state
    #         self.opt_state = self.optimizer.init(self.params)  # optimizer on the prior params
    #
    #
    #
    #     task_dicts = []
    #
    #     for xs,ys in meta_train_tuples:
    #         # state of a gp can encapsulate caches of already fitted data and hopefully speed up inference
    #         _, state = self._apply_fns.base_learner_fit(self.params, self.empty_state, self._rng, xs, ys)
    #         task_dict = {
    #             'xs_train': xs,
    #             'ys_train': ys,
    #             'hk_state': state
    #         }
    #
    #         if flatten_y:
    #             task_dict['ys_train'] = ys.flatten()
    #
    #         task_dicts.append(task_dict)
    #
    #     return task_dicts


if __name__ == "__main__":
    from jax.config import config
    config.update("jax_debug_nans", False)
    config.update('jax_disable_jit', False)

    from experiments.data_sim import SinusoidDataset

    data_sim = SinusoidDataset(random_state=np.random.RandomState(29))
    meta_train_data = data_sim.generate_meta_train_data(n_tasks=20, n_samples=10)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=50, n_samples_context=10, n_samples_test=160)

    NN_LAYERS = (32, 32, 32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """

    print('\n ---- GPR mll meta-learning ---- ')

    torch.set_num_threads(2)

    for weight_decay in [1.0]:
        pacoh_svgd = PACOH_SVGD_GP(1, 1, learning_mode='both', weight_decay=weight_decay, task_batch_size=2,
                                covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS, feature_dim=2,
                                kernel_nn_layers=NN_LAYERS)

        itrs = 0
        print("---- weight-decay =  %.4f ----"%weight_decay)

        for i in range(10):
            n_iter = 2000
            pacoh_svgd.meta_fit(meta_train_data, log_period=1000, num_iter_fit=n_iter)

            itrs += n_iter

            x_plot = np.linspace(-5, 5, num=150)
            x_context, y_context, x_test, y_test = meta_test_data[0]
            pred_mean, pred_std = pacoh_svgd.meta_predict(x_context, y_context, x_plot)
            # ucb, lcb = gp_model.confidence_intervals(x_context, x_plot)

            plt.scatter(x_test, y_test, color="green") # the unknown target test points
            plt.scatter(x_context, y_context, color="red") # the target train points
            plt.plot(x_plot, pred_mean)    # the curve we fitted based on the target test points

            pred_prior, _ = pacoh_svgd.meta_predict(jnp.zeros((0,1)), jnp.zeros((0,)), x_plot)
            plt.plot(x_plot, pred_prior.flatten(), color="red")
            lcb = pred_mean-pred_std
            ucb = pred_mean+pred_std
            plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.2, color="green")
            plt.title('GPR meta mll (weight-decay =  %.4f) itrs = %i' % (weight_decay, itrs))
            plt.show()