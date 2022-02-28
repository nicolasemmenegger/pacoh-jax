import functools
from typing import Callable, Collection, Union, Optional

import haiku as hk
import jax.random
import numpy as np
import optax
import torch
from jax import numpy as jnp
from numpyro.distributions import Uniform, MultivariateNormal

from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.models.pure.pure_functions import construct_gp_base_learner
from pacoh.modules.domain import ContinuousDomain, DiscreteDomain
from pacoh.modules.exact_gp import JAXExactGP
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.modules.means import JAXMean, JAXZeroMean
from pacoh.modules.kernels import JAXKernel, JAXRBFKernel
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.util.distributions import multivariate_kl
from pacoh.util.initialization import (
    initialize_model_with_state,
    initialize_optimizer,
    initialize_model,
)


class F_PACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        domain: Union[DiscreteDomain, ContinuousDomain],
        learning_mode: str = "both",
        weight_decay: float = 0.0,
        feature_dim: int = 2,
        covar_module: Union[str, Callable[[], JAXKernel]] = "NN",
        mean_module: Union[str, Callable[[], JAXMean]] = "NN",
        mean_nn_layers: Collection[int] = (32, 32),
        kernel_nn_layers: Collection[int] = (32, 32),
        task_batch_size: int = 5,
        num_tasks: int = None,
        lr: float = 1e-3,
        lr_decay: float = 1.0,
        prior_weight=0.1,  # kappa
        train_data_in_kl=False,  # whether to reuse the training set to evaluate the kl at
        num_samples_kl=20,
        hyperprior_lengthscale=0.3,
        hyperprior_outputscale=2.0,
        hyperprior_noise_var=1e-3,  # this has numerical stability implications
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        minibatch_at_dataset_level: bool = False, # if True, we can vectorize the loop ranging over the tasks
        dataset_batch_size: Optional[int] = None, # if None, and minibatch_at_dataset_level is True, then the datasets have to have the same size
        random_state: jax.random.PRNGKey = None,
    ):
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
            dataset_batch_size=dataset_batch_size
        )

        assert isinstance(domain, ContinuousDomain) or isinstance(domain, DiscreteDomain)
        assert domain.d == input_dim, "Domain and input dimension don't match"
        assert learning_mode in [
            "learn_mean",
            "learn_kernel",
            "both",
            "vanilla_gp",
        ], "Invalid learning mode"
        assert mean_module in ["NN", "constant", "zero"] or isinstance(
            mean_module, JAXMean
        ), "Invalid mean_module option"
        assert covar_module in ["NN", "SE"] or isinstance(
            covar_module, JAXKernel
        ), "Invalid covar_module option"
        """-------- Setup haiku differentiable functions and parameters -------"""
        pacoh_map_closure = construct_gp_base_learner(
            input_dim,
            output_dim,
            mean_module,
            covar_module,
            learning_mode,
            feature_dim,
            mean_nn_layers,
            kernel_nn_layers,
            learn_likelihood=True,
            initial_noise_std=1.0,
        )
        init_fn, self._apply_fns = hk.multi_transform_with_state(pacoh_map_closure)
        self._rng, init_key = jax.random.split(self._rng)
        self.particle, empty_state = initialize_model_with_state(
            init_fn, init_key, (task_batch_size, input_dim)
        )
        self.state = self.empty_state = empty_state

        # construct the hyperprior in function space
        def hyperprior_impure(xs):
            hyperprior_mean = JAXZeroMean(output_dim)
            hyperprior_covar = JAXRBFKernel(input_dim, hyperprior_lengthscale, hyperprior_outputscale)
            hyperprior_likelihood = JAXGaussianLikelihood(
                output_dim=output_dim,
                variance=0.0,  # does not matter, only used for prior
                learn_likelihood=False,
            )

            # the hyperprior is a dirac distribution over exactly one GP
            gp_hyperprior_particle = JAXExactGP(hyperprior_mean, hyperprior_covar, hyperprior_likelihood)
            return gp_hyperprior_particle.prior(xs, return_full_covariance=True)

        hyper_init, hyper_apply = hk.transform(hyperprior_impure)
        hyper_params = initialize_model(hyper_init, None, (1, input_dim))
        self.hyperprior_marginal = functools.partial(hyper_apply, hyper_params, None)
        self.hyperprior_noise_var = hyperprior_noise_var
        self._rng, sample_key = jax.random.split(self._rng)

        def target_post_prob_one_task(particle, key, xs, ys):
            mll, _ = self._apply_fns.base_learner_mll_estimator(particle, empty_state, None, xs, ys)
            kl = self._functional_kl(particle, key, xs)

            # this is formula (3) from pacoh, and prior_weight corresponds to kappa
            n = num_tasks
            m = xs.shape[0]
            return (
                    -mll / (task_batch_size * m)
                    + prior_weight * (1 / jnp.sqrt(n) + 1 / (n * m)) * kl / task_batch_size
            )

        def target_post_prob_loop(particle, key_for_sampling, meta_xs_batch, meta_ys_batch):
            """meta_xs_batch is a python list of jnp.arrays"""
            loss = 0.0
            keys = jax.random.split(key_for_sampling, len(meta_xs_batch))
            for xs, ys, key in zip(meta_xs_batch, meta_ys_batch, keys):
                loss += target_post_prob_one_task(particle, key, xs, ys)
            return loss

        target_post_prob_vmap_array = jax.vmap(target_post_prob_one_task, in_axes=(None, 0, 0, 0))
        def target_post_prob_vmap(particle, key_for_sampling, meta_xs_batch, meta_ys_batch):
            """meta_xs_batch is a python list of jnp.arrays"""
            keys = jax.random.split(key_for_sampling, len(meta_xs_batch))
            return jnp.sum(target_post_prob_vmap_array(particle, keys, meta_xs_batch, meta_ys_batch))

        if minibatch_at_dataset_level:
            self.target_val_and_grad = jax.jit(jax.value_and_grad(jax.jit(target_post_prob_vmap)))
        else:
            self.target_val_and_grad = jax.jit(jax.value_and_grad(target_post_prob_loop))

        # f-pacoh specific initialisation
        self.domain_dist = Uniform(low=domain.l, high=domain.u)
        self.train_data_in_kl = train_data_in_kl
        self.num_samples_kl = num_samples_kl

        # optimizer setup
        self.optimizer, self.optimizer_state = initialize_optimizer(
            "AdamW", lr, self.particle, lr_decay, weight_decay=weight_decay
        )

    def _meta_step(self, mini_batch) -> float:
        xs_batch, ys_batch = mini_batch
        self._rng, key = jax.random.split(self._rng)
        loss, grad = self.target_val_and_grad(self.particle, key, xs_batch, ys_batch)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, self.particle)
        self.particle = optax.apply_updates(self.particle, updates)
        return loss

    @normalize_predict
    def predict(self, xs):
        pred_dist, _ = self._apply_fns.base_learner_predict(self.particle, self.state, None, xs)
        return pred_dist

    def _recompute_posterior(self):
        # use the stored data in xs_data, ys_data to instantiate a base_learner
        _, self.state = self._apply_fns.base_learner_fit(
            self.particle, self.empty_state, None, self._xs_data, self._ys_data
        )

    def _sample_measurement_set(self, k, xs_train):
        if self.train_data_in_kl:
            raise NotImplementedError
            # n_train_x = min(xs_train.shape[0], self.num_samples_kl // 2)
            # n_rand_x = self.num_samples_kl - n_train_x
            # idx_rand = np.random.choice(xs_train.shape[0], n_train_x)
            # xs_kl = torch.cat([xs_train[idx_rand], self.domain_dist.sample((n_rand_x,))], dim=0)
        else:
            xs_kl = self.domain_dist.sample(k, (self.num_samples_kl,)) / self._normalizer.x_std[None, :]

        assert xs_kl.shape == (self.num_samples_kl, self.input_dim)
        return xs_kl

    def _functional_kl(self, particle, k, xs):
        """
        Computes the approximation of the functional kl divergence by subsampling
        :param particle: The parameters of the hyperposterior/prior
        :param xs: The dataset to fit
        :return:
        evaluate the kl of the predictive distributions of prior and posterior
        """
        # sample / construct measurement set
        xs_kl = self._sample_measurement_set(k, xs)

        hyperposterior, _ = self._apply_fns.base_learner_predict(particle, self.empty_state, None, xs_kl)
        hyperprior = self.hyperprior_marginal(xs_kl)

        assert isinstance(
            hyperposterior, MultivariateNormal
        ), "both hyperprior and hyperposterior should have same shape"

        def compute_kl_with_noise(noise):
            # computes the kl divergence between hyperprior and hyperposterior, by adding a little diagonal noise to the
            # hyperprior
            diagonal_noise = inject_noise_var * jnp.eye(self.num_samples_kl, self.num_samples_kl)
            covar_with_noise = hyperprior.covariance_matrix + diagonal_noise
            hyperprior_with_noise = MultivariateNormal(hyperprior.mean, covar_with_noise)
            kl = multivariate_kl(hyperposterior, hyperprior_with_noise)
            return kl

        # this is both differentiable and jittable
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow
        # standard for loop is differentiable as condition does not depend on data
        kl = jnp.nan
        inject_noise_var = self.hyperprior_noise_var
        for _ in range(5):
            # this
            kl = jax.lax.cond(
                jnp.isnan(kl),
                lambda n: compute_kl_with_noise(n),
                lambda _: kl,
                inject_noise_var,
            )
            inject_noise_var *= 2

        return kl

    def _clear_data(self):
        super()._clear_data()


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_debug_nans", False)
    config.update("jax_disable_jit", False)

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
        plt.title("sample from the GP prior")
        plt.show()
    """ 2) Classical mean learning based on mll """

    print("\n ---- GPR mll meta-learning ---- ")

    torch.set_num_threads(2)

    for weight_decay in [0.5]:
        pacoh_map = F_PACOH_MAP_GP(
            1,
            1,
            ContinuousDomain(jnp.ones((1,)) * -6, jnp.ones((1,)) * 6),
            learning_mode="both",
            weight_decay=weight_decay,
            task_batch_size=5,
            num_tasks=num_train_tasks,
            covar_module="NN",
            mean_module="NN",
            mean_nn_layers=NN_LAYERS,
            feature_dim=2,
            kernel_nn_layers=NN_LAYERS,
            minibatch_at_dataset_level=False,
        )

        itrs = 0
        print("---- weight-decay =  %.4f ----" % weight_decay)

        for i in range(40):
            n_iter = 500
            pacoh_map.meta_fit(meta_train_data, meta_test_data, log_period=1000, num_iter_fit=n_iter)

            itrs += n_iter

            x_plot = np.linspace(-5, 5, num=150)
            x_context, y_context, x_test, y_test = meta_test_data[1]

            # prior prediction
            pacoh_map._clear_data()
            prior_lcb, prior_ucb = pacoh_map.confidence_intervals(x_plot)
            prior_mean, prior_std = pacoh_map.predict(
                x_plot, return_density=False, return_full_covariance=False
            )

            # posterior prediction
            pred_dist = pacoh_map.meta_predict(x_context, y_context, x_plot, return_density=True)
            pred_mean = pred_dist.mean
            lcb, ucb = pacoh_map.confidence_intervals(x_plot)

            # plot data
            plt.scatter(x_test, y_test, color="green")  # the unknown target test points
            plt.scatter(x_context, y_context, color="red")  # the target train points

            # plot posterior
            plt.plot(x_plot, pred_mean, color="blue")  # the curve we fitted based on the target test points
            plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.15, color="blue")

            # plot prior
            plt.plot(
                x_plot, prior_mean, color="orange"
            )  # the curve we fitted based on the target test points
            plt.fill_between(
                x_plot,
                prior_lcb.flatten(),
                prior_ucb.flatten(),
                alpha=0.15,
                color="orange",
            )

            plt.title("GPR meta mll (weight-decay =  %.4f) itrs = %i" % (weight_decay, itrs))
            plt.show()
