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
from pacoh.util.data_handling import Statistics, handle_batch_input_dimensionality, DataNormalizer
from pacoh.modules.distributions import AffineTransformedDistribution
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel, pytree_rbf_set
from pacoh.util.initialization import initialize_batched_model
from pacoh.util.tree import pytrees_stack

from pacoh_map_gp import construct_pacoh_map_forward_fns, PACOH_MAP_GP
from pacoh.modules.batching import multi_transform_and_batch_module_with_state

# this is all there is to it
construct_meta_gp_forward_fns = multi_transform_and_batch_module_with_state(construct_pacoh_map_forward_fns)


class PACOH_SVGD_GP(RegressionModelMetaLearned):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 weight_prior_std=0.5,
                 bias_prior_std=3.0,
                 svgd_kernel = 'SVGD',
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
                 mean_nn_layers: Collection[int] = (32, 32),
                 kernel_nn_layers: Collection[int] = (32, 32),
                 task_batch_size: int = -1,
                 num_tasks: int = 1,
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
        init, self.apply, self.apply_broadcast, self.apply_manytomany = construct_meta_gp_forward_fns(
            input_dim, output_dim, mean_module, covar_module, 'both', feature_dim,
            mean_nn_layers, kernel_nn_layers, learn_likelihood)

        # c) initialize the the state of the hyperprior and of the hyperposterior particles
        self._rng, init_key = jax.random.split(self._rng)
        params, template = initialize_batched_model(init, num_particles, init_key, (self.task_batch_size, input_dim))

        def mean_std_map(mod_name: str, name: str, _: jnp.array):
            # note: there could also be a distrinction between kernel module and mean module I think here
            if LIKELIHOOD_MODULE_NAME in mod_name:
                return likelihood_prior_mean, likelihood_prior_std
            elif MLP_MODULE_NAME in mod_name:
                if MLP_WEIGHT_NAME in name:
                    return 0.0, weight_prior_std
                elif MLP_BIAS_NAME in bias_name:
                    return 0.0, bias_prior_std
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
        mll_many_many = jax.jit(jax.vmap(self.apply.base_learner_mll_estimator, (None, None, 0, 0), 1))
        self.svgd = SVGD(functools.partial(target_post_prob_batched, mll_many_many=mll_many_many),
                         jax.jit(kernel_fwd), self.optimizer, self.optimizer_state)

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        super().meta_fit(meta_train_tuples, meta_valid_tuples, verbose=verbose, log_period=log_period, n_iter=n_iter)
        """ Runs the meta train loop for some number of iterations """
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        # construct the loss function
        def regularized_loss(params, state, rng, xs, ys):
            ll, state = self._apply_fns.base_learner_mll_estimator(params, state, rng, xs, ys)
            prior = self.ll_under_hyperprior(params)
            return -ll - prior, state

        # batching
        batched_losses_and_states = jax.vmap(regularized_loss, in_axes=(None, 0, 0, 0, 0), out_axes=(0,0))
        def cumulative_loss(*args):
            lls, states = batched_losses_and_states(*args)
            return jnp.sum(lls), states



        # auto differentiation and jit compilation
        batched_mll_value_and_grad = jax.value_and_grad(cumulative_loss, has_aux=True)
        batched_mll_value_and_grad = jax.jit(batched_mll_value_and_grad)

        for itr in range(1, n_iter + 1):
            print(itr)
            # a) choose a minibatch. TODO: make this more elegant
            self._rng, choice_key = jax.random.split(self._rng)
            batch_indices = jax.random.choice(choice_key, len(task_dicts), shape=(self.task_batch_size,))
            batch = [task_dicts[i] for i in batch_indices]
            batched_xs = jnp.array([task['xs_train'] for task in batch])
            batched_ys = jnp.array([task['ys_train'] for task in batch])
            batched_states = pytrees_stack([task['hk_state'] for task in batch])
            apply_rngs = jax.random.split(self._rng, 1+self.task_batch_size)
            self._rng = apply_rngs[0]
            apply_rngs = apply_rngs[1:]

            # b) get value and gradient
            output, gradients = batched_mll_value_and_grad(self.params, batched_states, apply_rngs, batched_xs, batched_ys)
            loss, states = output  # we don't actually need the states


            # c) compute and apply optimizer updates
            updates, new_opt_state = self.optimizer.update(gradients, self.opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)

            cum_loss += loss

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                cum_loss = 0.0
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, avg_loss, duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if meta_valid_tuples is not None:
                    warnings.warn("implement validation option")
                    self.likelihood.eval()
                    valid_ll, valid_rmse, calibr_err, calibr_err_chi2 = self.eval_datasets(meta_valid_tuples)
                    self.likelihood.train()
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_ll, valid_rmse, calibr_err)

                if verbose:
                    logging.info(message)

        self.fitted = True

        warnings.warn("check what this assertion is used for")
        # assert self.X_data.shape[0] == 0 and self.y_data.shape[0] == 0, "Data for posterior inference can be passed " \
        #
        #                                                               "only after the meta-training"

        return cum_loss


    def meta_predict(self, context_x, context_y, test_x, return_density=False):
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
        """
        warnings.warn("I would suggest that meta_predict be deprecated in favor of separate fit and predict methods, no?", PendingDeprecationWarning)
        context_x, context_y = handle_batch_input_dimensionality(context_x, context_y)
        test_x = handle_batch_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)
        test_x = self._normalize_data(test_x, None)

        # note that we use the empty_state because that corresponds to no data, by construction
        self._rng, fit_key = jax.random.split(self._rng)
        _, fitted_state = self._apply_fns.base_learner_fit(self.params, self.empty_state, fit_key, context_x, context_y)
        self._rng, pred_key =  jax.random.split(self._rng)
        pred_dist, state = self._apply_fns.base_learner_predict(self.params, fitted_state, pred_key, test_x)
        pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                             normalization_std=self.y_std)

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean, pred_std

    def _recompute_posterior(self):
        # use the stored data in xs_data, ys_data to instantiate a base_learner
        self._rng, fitkey = self._rng.split()
        self._apply_fns.base_learner_fit(self.params,
                                         fitkey,
                                         self.xs_data, self.ys_data)


    # def predict(self, test_x, return_density=False):
    #     test_x = _handle_batch_input_dimensionality(test_x)
    #     test_x = self.normalize_data(test_x)
    #     pred_dist = self._apply_fns.base_learner_predict(self._updater_state['params'], self._rng, test_x)
    #
    #     pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
    #                                                           normalization_std=self.y_std)
    #
    #     if return_density:
    #         return pred_dist_transformed
    #     else:
    #         pred_mean = pred_dist_transformed.mean
    #         pred_std = pred_dist_transformed.stddev
    #         return pred_mean, pred_std
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
        pacoh_map = PACOH_MAP_GP(1, learning_mode='both', num_iter_fit=20000, weight_decay=weight_decay, task_batch_size=2,
                                covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS, feature_dim=2,
                                kernel_nn_layers=NN_LAYERS)

        itrs = 0
        print("---- weight-decay =  %.4f ----"%weight_decay)

        for i in range(10):
            n_iter = 2000
            pacoh_map.meta_fit(meta_train_data, log_period=1000, n_iter=n_iter)

            itrs += n_iter

            x_plot = np.linspace(-5, 5, num=150)
            x_context, y_context, x_test, y_test = meta_test_data[0]
            pred_mean, pred_std = pacoh_map.meta_predict(x_context, y_context, x_plot)
            # ucb, lcb = gp_model.confidence_intervals(x_context, x_plot)

            plt.scatter(x_test, y_test, color="green") # the unknown target test points
            plt.scatter(x_context, y_context, color="red") # the target train points
            plt.plot(x_plot, pred_mean)    # the curve we fitted based on the target test points

            pred_prior, _ = pacoh_map.meta_predict(jnp.zeros((0,1)), jnp.zeros((0,)), x_plot)
            plt.plot(x_plot, pred_prior.flatten(), color="red")
            lcb = pred_mean-pred_std
            ucb = pred_mean+pred_std
            plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.2, color="green")
            plt.title('GPR meta mll (weight-decay =  %.4f) itrs = %i' % (weight_decay, itrs))
            plt.show()