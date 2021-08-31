import warnings

import numpyro.distributions
import optax
import torch
import gpytorch
import time
import numpy as np
from absl import logging
from jax import numpy as jnp

from meta_bo.models.base.gp_components import LearnedGPRegressionModel, JAXConstantMean, JAXMean, JAXZeroMean
from meta_bo.models.base.kernels import JAXRBFKernel, JAXRBFKernelNN, JAXKernel
from meta_bo.models.base.neural_network import NeuralNetwork, JAXNeuralNetwork
from meta_bo.models.base.distributions import AffineTransformedDistribution, JAXGaussianLikelihood
from meta_bo.models.util import _handle_input_dimensionality
from meta_bo.models.abstract import RegressionModelMetaLearned
from config import device

class PACOH_MAP_GP(RegressionModelMetaLearned):

    def __init__(self,
                 input_dim,
                 learning_mode='both',
                 weight_decay=0.0,
                 feature_dim=2,
                 num_iter_fit=10000,
                 covar_module='NN',
                 mean_module='NN',
                 mean_nn_layers=(32, 32),
                 kernel_nn_layers=(32, 32),
                 task_batch_size=5,
                 lr=1e-3,
                 lr_decay=1.0,
                 normalize_data=True,
                 normalization_stats=None,
                 random_seed=None):

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
        self.normalize_data = normalize_data

        """ Setup prior, likelihood and optimizer """
        # note there is only one prior, because there is only one particle
        self._setup_gp_prior(mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers)
        self.likelihood = JAXGaussianLikelihood(variance_constraint_gt=1e-3)
        self._setup_optimizer(lr, lr_decay)

        warnings.warn("do something with shared parameters, or check if haiku does everything I need ")
        """ ------- normalization stats & data setup  ------- """
        self._normalization_stats = normalization_stats
        self.reset_to_prior()
        self.fitted = False

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        for itr in range(1, n_iter + 1):
            # actual meta-training step
            warnings.warn("self._rds.choice not correct")
            task_dict_batch = task_dicts[:self.task_batch_size] # self._rds.choice(task_dicts, size=self.task_batch_size)

            loss = self._step(task_dict_batch)
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

        # set gpytorch modules to eval mode and set gp to meta-learned gp prior
        assert self.X_data.shape[0] == 0 and self.y_data.shape[0] == 0, "Data for posterior inference can be passed " \
                                                                        "only after the meta-training"
        for task_dict in task_dicts:
            task_dict['model'].eval()
        self.likelihood.eval()
        self.reset_to_prior()
        return loss

    def predict(self, test_x, return_density=False, **kwargs):
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).float().to(device)

            pred_dist = self.likelihood(self.gp(test_x_tensor))
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)
            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std

    def predict_mean_std(self, test_x):
        return self.predict(test_x, return_density=False)

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

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(context_x, context_y, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module)
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x))
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.cpu().numpy(), pred_std.cpu().numpy()

    def reset_to_prior(self):
        warnings.warn("reimplement reset_to_prior")
        self.gp = LearnedGPRegressionModel(self.mean_module, self.covar_module, self.likelihood)
        # self._reset_data()
        # self.gp = lambda x: self._prior(x)

    def _recompute_posterior(self):
        x_context = torch.from_numpy(self.X_data).float().to(device)
        y_context = torch.from_numpy(self.y_data).float().to(device)
        self.gp = LearnedGPRegressionModel(x_context, y_context, self.likelihood,
                                      learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                      covar_module=self.covar_module, mean_module=self.mean_module)

    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.task_dicts[0]['model'].state_dict()
        }
        for task_dict in self.task_dicts:
            for key, tensor in task_dict['model'].state_dict().items():
                assert torch.all(state_dict['model'][key] == tensor).item()
        return state_dict

    def load_state_dict(self, state_dict):
        for task_dict in self.task_dicts:
            task_dict['model'].load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _prior(self, x):
        warnings.warn("here I  need an haiku transform and with a forward function. I also need to keep the params somewhere that haiku init gives me")
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        # if self.nn_kernel_map is not None:
        #     projected_x = self.nn_kernel_map(x)
        # else:
        #     projected_x = x
        #
        #     # feed through mean module
        # if self.nn_mean_fn is not None:
        #     mean_x = self.nn_mean_fn(x).squeeze()
        # else:
        #     mean_x = self.mean_module(projected_x).squeeze()
        #
        # covar_x = self.covar_module(projected_x)
        # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _prepare_meta_train_tasks(self, meta_train_tuples):
        self._check_meta_data_shapes(meta_train_tuples)

        if self._normalization_stats is None:
            self._compute_meta_normalization_stats(meta_train_tuples)
        else:
            self._set_normalization_stats(self._normalization_stats)

        task_dicts = [self._dataset_to_task_dict(x, y) for x, y in meta_train_tuples]
        return task_dicts

    def _dataset_to_task_dict(self, x, y):
        # a) prepare data
        xs, ys = self._prepare_data_per_task(x, y)
        task_dict = {'train_x': xs, 'train_y': ys}

        # b) prepare model and fit it to task
        task_dict['model'] = LearnedGPRegressionModel(self.mean_module, self.covar_module, self.likelihood)
        warnings.warn("this should probably be decoupled")
        task_dict['model'].fit(xs, ys)
        task_dict['mll_fn'] = lambda _, __:  task_dict['model'].pred_ll()
        return task_dict

    def _get_batch_loss(self, task_dicts):
        # this function will look completely different
        assert len(task_dicts) > 0
        # self.optimizer.zero_grad()
        def single_task_loss(task_dict):
            output = task_dict['model'](task_dict['train_x'])
            mll = task_dict['mll_fn'](output, task_dict['train_y'])
            return -mll

        all_losses = vmap(single_task_loss)
        return jnp.sum(all_losses(task_dicts))

        # loss.backward()
        # self.optimizer.step()
        # self.lr_scheduler.step()
        # return loss.item()

    def _setup_gp_prior(self, mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers):
        # setup kernel module
        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            self.covar_module = JAXRBFKernelNN(self.input_dim, feature_dim, layer_sizes=kernel_nn_layers)
        elif covar_module == 'SE':
            self.covar_module = JAXRBFKernel(input_dim=self.input_dim)
        elif isinstance(covar_module, JAXKernel):
            self.covar_module = covar_module
        else:
            raise ValueError('Invalid covar_module option')

        # setup mean module
        if mean_module == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            self.mean_module = JAXNeuralNetwork(input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers)
        elif mean_module == 'constant':
            self.mean_module = JAXConstantMean()
        elif mean_module == 'zero':
            self.mean_module = JAXZeroMean()
        elif isinstance(mean_module, JAXMean):
            self.mean_module = mean_module
        else:
            raise ValueError('Invalid mean_module option')

            # c) add parameters of covar and mean module if desired
        warnings.warn("Check if I have the right lr scheduler for the different parts")
        # # c) add parameters of covar and mean module if desired
        # if learning_mode in ["learn_kernel", "both"]:
        #     self.shared_parameters.append({'params': self.covar_module.hyperparameters(), 'lr': self.lr})
        #
        # if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
        #     self.shared_parameters.append({'params': self.mean_module.hyperparameters(), 'lr': self.lr})



    def _setup_optimizer(self, lr: float, lr_decay: float):
        """
        Sets up the optimizer AdamW

        Args:
            lr: the initial learning rate
            lr_decay: the decay to apply to the learning rate every 1000 steps (not epochs, right?) TODO check
        """
        if lr_decay < 1.0:
            # staircase = True means it's the same as StepLR from torch.optim
            self.lr_scheduler = optax.exponential_decay(lr, 1000, decay_rate=lr_decay, staircase=True)
        else:
            self.lr_scheduler = optax.constant_schedule(lr)

        self.optimizer = optax.adamw(self.lr_scheduler, weight_decay=self.weight_decay)
        warnings.warn("we need to have different weight-decays for some of the parameters!! -> regularization of log-scale parameters like the likelihood variance is problematic otherwise")
        warnings.warn("check that the shared-parameters that the torch.optimizer works on are the same. Also check that you even need self.lr_scheduler (i.e. does it need to be a class property")

    def _vectorize_pred_dist(self, pred_dist: numpyro.distributions.Distribution):
        """
        Models the predictive distribution passed according to an independent, heteroscedastic Gaussian,
        i.e. forgets about covariance in case the distribution was multivariate.
        """
        return torch.distributions.Normal(pred_dist.mean, pred_dist.scale)


if __name__ == "__main__":
    from experiments.data_sim import GPFunctionsDataset, SinusoidDataset

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
        pacoh_map = PACOH_MAP_GP(1, num_iter_fit=20000, weight_decay=weight_decay, task_batch_size=2,
                                covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
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