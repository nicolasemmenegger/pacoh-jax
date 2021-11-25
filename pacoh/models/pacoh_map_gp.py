import functools
import time
import warnings
from typing import Callable, Collection, Union

import haiku as hk
import jax.random
import numpy as np
import numpyro.distributions
import optax
import torch
from absl import logging
from jax import numpy as jnp

from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.models.pacoh_svgd_gp import PACOH_SVGD_GP
from pacoh.models.pure.pure_functions import construct_pacoh_map_forward_fns
from pacoh.util.data_handling import DataNormalizer, handle_batch_input_dimensionality
from pacoh.modules.distributions import AffineTransformedDistribution
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel
from pacoh.util.tree import pytrees_stack

class PACOH_MAP_GP(PACOH_SVGD_GP):
    def __init__(self, *args, **kwargs):
        kwargs['num_particles'] = 1
        super().__init__(*args, kwargs)
        # think of how I could do this here

        # super.= partial(super().__init__, num_particles=1)

class PACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 learning_mode: str = 'both',
                 weight_decay: float = 0.0,
                 feature_dim: int = 2,
                 covar_module: Union[str, Callable[[], JAXKernel]] = 'NN',
                 mean_module: Union[str, Callable[[], JAXMean]] = 'NN',
                 mean_nn_layers: Collection[int] = (32, 32),
                 kernel_nn_layers: Collection[int] = (32, 32),
                 task_batch_size: int = 5,
                 lr: float = 1e-3,
                 lr_decay: float = 1.0,
                 normalize_data: bool = True,
                 normalizer: DataNormalizer = None,
                 random_state: jax.random.PRNGKey = None):
        """
        :param input_dim:
        :param learning_mode:
        :param weight_decay:
        :param feature_dim:
        :param num_iter_fit:
        :param covar_module:
        :param mean_module:
        :param mean_nn_layers:
        :param kernel_nn_layers:
        :param task_batch_size:
        :param lr:
        :param lr_decay:
        :param normalize_data:
        :param normalization_stats:
        :param random_seed:
        """
        super().__init__(input_dim, output_dim, normalize_data, normalizer, random_state)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla_gp'], 'Invalid learning mode'
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, JAXMean), 'Invalid mean_module option'
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, JAXKernel), 'Invalid covar_module option'

        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_dim = feature_dim
        self.task_batch_size = task_batch_size
        self.mean_nn_layers = mean_nn_layers
        self.kernel_nn_layers = kernel_nn_layers

        """ Optimizer setup """
        # note there is only one prior, because there is only one particle
        if lr_decay < 1.0:
            # staircase = True means it's the same as StepLR from torch.optim
            self.lr_scheduler = optax.exponential_decay(lr, 1000, decay_rate=lr_decay, staircase=True)
        else:
            self.lr_scheduler = optax.constant_schedule(lr)

        """ ------- normalization stats & data setup  ------- """
        self.fitted = False

        """-------- Setup haiku differentiable functions and parameters -------"""
        pacoh_map_closure = construct_pacoh_map_forward_fns(input_dim, mean_module, covar_module, learning_mode,
                                                            feature_dim, mean_nn_layers, kernel_nn_layers)
        self._init_fn, self._apply_fns = hk.multi_transform_with_state(pacoh_map_closure)
        # jit compile the pure mll function, as this is the bottleneck
        self._apply_fns = self._apply_fns._replace(
            base_learner_mll_estimator=jax.jit(self._apply_fns.base_learner_mll_estimator)
        )

        # mask weight decay for log_scale parameters
        self.mask_fn = functools.partial(hk.data_structures.map, lambda _, name, __: name != '__positive_log_scale_param')
        self.optimizer = optax.adamw(self.lr_scheduler, weight_decay=self.weight_decay, mask=self.mask_fn)
        self.prior_params: hk.Params
        self.opt_state: optax.OptState

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, log_period=500, n_iter=None):
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

            # gradients = hk.data_structures.map(lambda _,__,x: 0.0, gradients) # zerograd

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

    def ll_under_hyperprior(self, params):
        # note to future me https://github.com/google/jax/issues/2962
        # explains why you can't index arrays from jax.lax.fori_loop's body, which is very confusing at first
        def map_fn(_, name, data):
            if name == "__positive_log_scale_param":
                data = jnp.exp(data)

            if jnp.isscalar(data):
                return jnp.array([data])
            else:
                return data.flatten()
        mapped = hk.data_structures.map(map_fn, params)
        leaves = jax.tree_leaves(mapped)
        all_params = jnp.concatenate(leaves)
        normal = numpyro.distributions.Normal()
        log_probs = normal.log_prob(all_params)
        return jnp.sum(log_probs)

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
        _, fitted_state = self.apply.base_learner_fit(self.params, self.empty_state, fit_key, context_x, context_y)
        self._rng, pred_key =  jax.random.split(self._rng)
        pred_dist, state = self.apply.base_learner_predict(self.params, fitted_state, pred_key, test_x)
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
                                         self._xs_data, self._ys_data)


    def predict(self, test_x, return_density=False, **kwargs):
        test_x = handle_batch_input_dimensionality(test_x)
        test_x = self._normalize_data(test_x)
        pred_dist = self._apply_fns.base_learner_predict(self._updater_state['params'], self._rng, test_x)

        pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                              normalization_std=self.y_std)

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean, pred_std



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