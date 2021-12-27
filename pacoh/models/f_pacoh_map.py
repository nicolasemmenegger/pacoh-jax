import functools
import sys
import time
import warnings
from typing import Callable, Collection, Union

import haiku as hk
import jax.random
import numpy as np
import optax
import torch
from jax import numpy as jnp
import numpyro.distributions
from numpyro.distributions import Uniform, MultivariateNormal, Independent

from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.models.pure.pure_functions import construct_gp_base_learner
from pacoh.modules.belief import GaussianBelief, GaussianBeliefState
from pacoh.modules.domain import ContinuousDomain, DiscreteDomain
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel
from pacoh.modules.distributions import multivariate_kl
from pacoh.util.initialization import initialize_model_with_state, initialize_optimizer


class F_PACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 domain: Union[DiscreteDomain, ContinuousDomain],
                 learning_mode: str = 'both',
                 weight_decay: float = 0.0,
                 feature_dim: int = 2,
                 covar_module: Union[str, Callable[[], JAXKernel]] = 'NN',
                 mean_module: Union[str, Callable[[], JAXMean]] = 'NN',
                 mean_nn_layers: Collection[int] = (32, 32),
                 kernel_nn_layers: Collection[int] = (32, 32),
                 task_batch_size: int = 5,
                 num_tasks: int = None,
                 lr: float = 1e-3,
                 lr_decay: float = 1.0,
                 prior_weight=1e-3,
                 train_data_in_kl=False, # whether to reuse the training set to evaluate the kl at
                 num_samples_kl=20,
                 prior_lengthscale=0.2,
                 prior_outputscale=2.0,
                 prior_kernel_noise=1e-3,
                 normalize_data: bool = True,
                 normalizer: DataNormalizer = None,
                 random_state: jax.random.PRNGKey = None):
        super().__init__(input_dim, output_dim, normalize_data, normalizer, random_state, task_batch_size, num_tasks)

        assert isinstance(domain, ContinuousDomain) or isinstance(domain, DiscreteDomain)
        assert domain.d == input_dim, "Domain and input dimension don't match"
        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla_gp'], 'Invalid learning mode'
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, JAXMean), 'Invalid mean_module option'
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, JAXKernel), 'Invalid covar_module option'

        """-------- Setup haiku differentiable functions and parameters -------"""
        pacoh_map_closure = construct_gp_base_learner(input_dim, output_dim, mean_module, covar_module,
                                                      learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers,
                                                      learn_likelihood=True, initial_noise_std=0.05)
        init_fn, self._apply_fns = hk.multi_transform_with_state(pacoh_map_closure)
        self._rng, init_key = jax.random.split(self._rng)
        self.particle, empty_state = initialize_model_with_state(init_fn, init_key, (task_batch_size, input_dim))
        self.state = self.empty_state = empty_state

        def mean_std_map(*_):
            return 0.0, 1.0

        self.hyperprior = GaussianBeliefState.initialize_heterogenous(mean_std_map, self.particle)

        self._rng, sample_key = jax.random.split(self._rng)
        # self.particle = pytree_unstack(GaussianBelief.rsample(self.hyperprior, sample_key, 1))[0]


        def target_post_prob(particle, key_for_sampling, meta_xs_batch, meta_ys_batch):
            """ Assumes meta_xs is a list of len = task_batch_size, with each element being an array of a variable
            number of elements of shape [input_size]. Similarly for meta_ys
            """
            loss = 0.0
            keys = jax.random.split(key_for_sampling, len(meta_xs_batch))
            for xs, ys, k in zip(meta_xs_batch, meta_ys_batch, keys):
                # marginal ll
                mll, _ = self._apply_fns.base_learner_mll_estimator(particle, empty_state, None, xs, ys)
                kl = self._functional_kl(particle, None, k, xs, ys)
                n = num_tasks
                m = xs.shape[0]
                loss += - mll / (task_batch_size * m) + prior_weight * (1 / jnp.sqrt(n) + 1 / (n * m)) * kl / task_batch_size

            return loss

        self.target = target_post_prob

        self.target_val_and_grad = jax.jit(jax.value_and_grad(target_post_prob))

        # f-pacoh specific initialisation
        self.domain_dist = Uniform(low=domain.l, high=domain.u)
        self.train_data_in_kl = train_data_in_kl
        self.num_samples_kl = num_samples_kl

        # optimizer setup
        self.optimizer, self.optimizer_state = initialize_optimizer("AdamW", lr, self.particle,
                                                                    lr_decay, weight_decay=weight_decay)

    def _meta_step(self, mini_batch) -> float:
        xs_batch, ys_batch = mini_batch
        self._rng, key = jax.random.split(self._rng)
        loss, grad = self.target_val_and_grad(self.particle, key, xs_batch, ys_batch)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, self.particle)
        self.particle = optax.apply_updates(self.particle, updates)
        if jnp.isnan(loss):
            print(loss)
            keys = jax.random.split(key, 5)
            for i in range(5):
                print(self._sample_measurement_set(keys[i], xs_batch))
                # self._apply_fns.base_learner_predict(self.particle, )
            sys.exit(1)
        return loss


    @normalize_predict
    def predict(self, xs):
        pred_dist, _ = self._apply_fns.base_learner_predict(self.particle, self.state, None, xs)
        return pred_dist

    def _recompute_posterior(self):
        # use the stored data in xs_data, ys_data to instantiate a base_learner
        _, self.state = self._apply_fns.base_learner_fit(self.particle,
                                                         self.empty_state,
                                                         None,
                                                         self._xs_data,
                                                         self._ys_data)

    def _sample_measurement_set(self, k, xs_train):
        if self.train_data_in_kl:
            raise NotImplementedError
            # n_train_x = min(xs_train.shape[0], self.num_samples_kl // 2)
            # n_rand_x = self.num_samples_kl - n_train_x
            # idx_rand = np.random.choice(xs_train.shape[0], n_train_x)
            # xs_kl = torch.cat([xs_train[idx_rand], self.domain_dist.sample((n_rand_x,))], dim=0)
        else:
            xs_kl = self.domain_dist.sample(k, (self.num_samples_kl,))

        assert xs_kl.shape == (self.num_samples_kl, self.input_dim)
        return xs_kl

    def _functional_kl(self, particle, state, k, xs, ys):
        """
        Computes the approximation of the functional kl divergence by subsampling
        :param particle: The parameters of the hyperposterior/prior
        :param state: The state of the fitted model from estimating the mll
        :param xs: The dataset to fit
        :param ys: The dataset to fit
        :return:
        evaluate the kl of the predictive distributions of prior and posterior
        """
        # sample / construct measurement set
        xs_kl = self._sample_measurement_set(k, xs)

        # functional KL
        # TODO check that this is really not needed anymore, if no, can get rid of ys argument
        _, fitted_state = self._apply_fns.base_learner_fit(particle, self.empty_state, None, xs, ys)
        pred_posterior, _ = self._apply_fns.base_learner_predict(particle, fitted_state, None, xs_kl)
        pred_prior, _ = self._apply_fns.base_learner_predict(particle, self.empty_state, None, xs_kl)

        warnings.warn("I should implement an option to return the full posterior")


        # multivar_posterior = distributions.Independent(pred_posterior, len(pred_posterior.batch_shape))
        # multivar_prior = distributions.Independent(pred_posterior, len(pred_prior.batch_shape))

        if isinstance(pred_prior, MultivariateNormal):
            assert isinstance(pred_posterior, MultivariateNormal), "both prior and posterior should have same shape"
            return multivariate_kl(pred_posterior, pred_prior)
        else:
            assert isinstance(pred_posterior, Independent) and isinstance(pred_prior, Independent)
            return numpyro.distributions.kl_divergence(pred_posterior, pred_prior)



        # K_prior = torch.reshape(self.prior_covar_module(x_kl).evaluate(), (x_kl.shape[0], x_kl.shape[0]))
        #
        # inject_noise_std = self.prior_kernel_noise
        # error_counter = 0
        # while error_counter < 5:
        #     try:
        #         dist_f_prior = MultivariateNormal(torch.zeros(x_kl.shape[0]), K_prior + inject_noise_std * torch.eye(x_kl.shape[0]))
        #         return kl_divergence(dist_f_posterior, dist_f_prior)
        #     except RuntimeError as e:
        #         import warnings
        #         inject_noise_std = 2 * inject_noise_std
        #         error_counter += 1
        #         warnings.warn('encoundered numerical error in computation of KL: %s '
        #                       '--- Doubling inject_noise_std to %.4f and trying again' % (str(e), inject_noise_std))
        # raise RuntimeError('Not able to compute KL')


if __name__ == "__main__":

    dist1 = numpyro.distributions.Normal(loc=jnp.zeros((3,)), scale=1.1*jnp.ones((3,)))
    dist2 = numpyro.distributions.Normal(loc=0.1+jnp.zeros((3,)), scale=0.9*jnp.ones((3,)))

    dist1 = numpyro.distributions.Independent(dist1, 0)
    dist2 = numpyro.distributions.Independent(dist2, 0)

    print(numpyro.distributions.kl_divergence(dist1, dist2))

    from jax.config import config
    config.update("jax_debug_nans", False)
    config.update('jax_disable_jit', False)

    from experiments.data_sim import SinusoidDataset

    data_sim = SinusoidDataset(random_state=np.random.RandomState(29))
    num_train_tasks = 20
    meta_train_data = data_sim.generate_meta_train_data(n_tasks=num_train_tasks, n_samples=10)
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

    for weight_decay in [0.5]:
        pacoh_map = F_PACOH_MAP_GP(1, 1, ContinuousDomain(jnp.ones((1,))*-6, jnp.ones((1,))*6), learning_mode='both', weight_decay=weight_decay, task_batch_size=5,
                                 num_tasks=num_train_tasks,
                                covar_module='NN', mean_module='zero', mean_nn_layers=NN_LAYERS, feature_dim=2,
                                kernel_nn_layers=NN_LAYERS)

        itrs = 0
        print("---- weight-decay =  %.4f ----"%weight_decay)

        for i in range(40):
            n_iter = 500
            pacoh_map.meta_fit(meta_train_data, meta_test_data, log_period=1000, num_iter_fit=n_iter)

            itrs += n_iter

            x_plot = np.linspace(-5, 5, num=150)
            x_context, y_context, x_test, y_test = meta_test_data[0]
            pred_dist = pacoh_map.meta_predict(x_context, y_context, x_plot, return_density=True)
            pacoh_map._clear_data()
            pacoh_map.predict(x_plot, False)
            pred_mean = pred_dist.loc

            plt.scatter(x_test, y_test, color="green")  # the unknown target test points
            plt.scatter(x_context, y_context, color="red")  # the target train points
            plt.plot(x_plot, pred_mean)    # the curve we fitted based on the target test points

            lcb, ucb = pacoh_map.confidence_intervals(x_plot)
            plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.2, color="green")
            plt.title('GPR meta mll (weight-decay =  %.4f) itrs = %i' % (weight_decay, itrs))
            plt.show()