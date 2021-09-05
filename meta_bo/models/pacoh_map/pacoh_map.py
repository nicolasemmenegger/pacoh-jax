import functools
import time
import warnings
from typing import Callable, NamedTuple, Tuple, Collection, Union

import haiku as hk
import jax.random
import numpy as np
import optax
import torch
from absl import logging
from jax import numpy as jnp

from meta_bo.models.base.abstract import RegressionModelMetaLearned
from meta_bo.models.base.data_handling import Statistics
from meta_bo.models.base.distributions import AffineTransformedDistribution
from meta_bo.models.base.distributions import JAXGaussianLikelihood
from meta_bo.models.base.gp_components import JAXExactGP, JAXMean, JAXConstantMean, JAXZeroMean
from meta_bo.models.base.gradient_updater import GradientUpdater
from meta_bo.models.base.kernels import JAXRBFKernelNN, JAXRBFKernel, JAXKernel
from meta_bo.models.base.neural_network import JAXNeuralNetwork
from meta_bo.models.util import _handle_batch_input_dimensionality


class BaseLearnerInterface(NamedTuple):
    """TODO this needs to be vectorized in some way. The MAP version needs to share the same cholesky accross calls
    kernel, mean and likelihood, but needs to perform target inference in parallel. """
    """This is the interface PACOH base learners should provide.
    hyper_prior_ll: A function that yields the log likelihood of the prior parameters under the hyperprior
    base_learner_fit: Fits the base learner to some data # maybe I need state here
    base_learner_predict: Actual predict on a task
    base_learner_mll_estimator: The mll of the base estimator under the data one just passed it
    """
    base_learner_fit: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], None]
    base_learner_predict: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], None]
    base_learner_fit_and_predict: Callable[[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], jnp.ndarray]
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
        base_learner = JAXExactGP(mean_module, covar_module, likelihood)

        init_fn = base_learner.pred_dist
        base_learner_fit = base_learner.fit
        base_learner_predict = base_learner.pred_dist

        def base_learner_fit_and_predict(xs, ys, xs_test):
            base_learner.fit(xs, ys)
            return base_learner.pred_dist(xs_test)

        def base_learner_mll_estimator(xs, ys):
            base_learner.fit(xs, ys)
            return base_learner.marginal_ll()

        # this is the interface I want to vmap probably
        return init_fn, BaseLearnerInterface(base_learner_fit=base_learner_fit,
                                             base_learner_predict=base_learner_predict,
                                             base_learner_mll_estimator=base_learner_mll_estimator,
                                             base_learner_fit_and_predict=base_learner_fit_and_predict)

    return factory

class PACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(self,
                 input_dim: int,
                 learning_mode: str = 'both',
                 weight_decay: float = 0.0,
                 feature_dim: int = 2,
                 num_iter_fit: int =10000,
                 covar_module: Union[str, Callable[[], JAXKernel]] = 'NN',
                 mean_module: Union[str, Callable[[], JAXMean]] ='NN',
                 mean_nn_layers: Collection[int] = (32, 32),
                 kernel_nn_layers: Collection[int] = (32, 32),
                 task_batch_size: int = 5,
                 lr: float = 1e-3,
                 lr_decay: float = 1.0,
                 normalize_data: bool = True,
                 normalization_stats: Statistics = None,
                 random_seed: jax.random.PRNGKey = None):
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
        super().__init__(normalize_data, random_seed)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla'], 'Invalid learning mode'
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, JAXMean), 'Invalid mean_module option'
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, JAXKernel), 'Invalid covar_module option'

        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_dim = feature_dim
        self.num_iter_fit = num_iter_fit
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
        self._normalization_stats = normalization_stats
        # self.reset_to_prior()
        self.fitted = False

        """-------- Setup haiku differentiable functions and parameters -------"""
        pacoh_map_closure = construct_pacoh_map_forward_fns(input_dim, mean_module, covar_module, learning_mode,
                                                         feature_dim, mean_nn_layers, kernel_nn_layers)
        self._init_fn, self._apply_fns = hk.multi_transform_with_state(pacoh_map_closure)
        mask_fn = functools.partial(jax.tree_map, lambda _: True)
        self.optimizer = optax.adamw(self.lr_scheduler, weight_decay=self.weight_decay, mask=mask_fn)
        self.updater = GradientUpdater(self._init_fn, self._apply_fns.base_learner_mll_estimator, self.optimizer)

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        """ Runs the meta train loop for some number of iterations """
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)

        if not self.fitted:
            # we have to initialize our updater, passing it some template data
            warnings.warn("I think I'd prefer to have that same as in vae.py in the official haiku codebase")
            self._updater_state = self.updater.init(self._rds, *task_dicts[0])
            self.fitted = True

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        for itr in range(1, n_iter + 1):
            # actual meta-training step
            task_dict = task_dicts[0] # can I use the jax random key here? # self._rds.choice(task_dicts, size=self.task_batch_size)

            warnings.warn("implement batching via vmap, maybe inside the updater")
            # returns the new parameters based on one task
            # what is self._updater_state at this point?
            self._updater_state, loss = self.updater.update(self._updater_state, *task_dict)
            cum_loss += loss['negative_mle']

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
        #                                                                "only after the meta-training"

        
        # for task_dict in task_dicts:
        #     task_dict['model'].eval()
        # self.likelihood.eval()
        # self.reset_to_prior()
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
        context_x, context_y = _handle_batch_input_dimensionality(context_x, context_y)
        test_x = _handle_batch_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)
        test_x = self._normalize_data(test_x, None)

        pred_dist, self._updater_state['hk_state'] = self._apply_fns.base_learner_fit_and_predict(self._updater_state['params'], self._updater_state['hk_state'], self._rds, context_x, context_y, test_x)
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
        self._apply_fns.base_learner_fit(self._updater_state['params'],
                                         self._rds,
                                         (self.xs_data, self.ys_data))


    def predict(self, test_x, return_density=False):
        test_x = _handle_batch_input_dimensionality(test_x)
        test_x = self.normalize_data(test_x)
        pred_dist = self._apply_fns.base_learner_predict(self._updater_state['params'], self._rds, test_x)

        pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                              normalization_std=self.y_std)

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean, pred_std


if __name__ == "__main__":
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

    for weight_decay in [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]:
        pacoh_map = PACOH_MAP_GP(1, learning_mode='both', num_iter_fit=20000, weight_decay=weight_decay, task_batch_size=2,
                                covar_module='SE', mean_module='constant', mean_nn_layers=NN_LAYERS, feature_dim=2,
                                kernel_nn_layers=NN_LAYERS)



        itrs = 0
        print("---- weight-decay =  %.4f ----"%weight_decay)

        for i in range(10):
            pacoh_map.meta_fit(meta_train_data, log_period=1000, n_iter=2000)

            itrs += 2000

            x_plot = np.linspace(-5, 5, num=150)
            x_context, t_context, x_test, y_test = meta_test_data[0]
            pred_mean, pred_std = pacoh_map.meta_predict(x_context, t_context, x_plot)
            # ucb, lcb = gp_model.confidence_intervals(x_context, x_plot)

            plt.scatter(x_test, y_test)
            plt.scatter(x_context, t_context)

            plt.plot(x_plot, pred_mean)
            # plt.fill_between(x_plot, lcb, ucb, alpha=0.2)
            plt.title('GPR meta mll (weight-decay =  %.4f) itrs = %i' % (weight_decay, itrs))
            plt.show()