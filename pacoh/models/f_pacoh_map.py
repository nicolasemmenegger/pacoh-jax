import functools
from typing import Callable, Collection, Union, Optional

import haiku as hk
import jax.random
import numpy as np
import optax
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
        prior_factor: float = 0.1,
        train_data_in_kl: bool =True,
        num_samples_kl: int = 20,
        hyperprior_lengthscale: float = 0.2,  # prior_lengthscale
        hyperprior_outputscale: float = 2.0,  # prior_outputscale
        hyperprior_noise_std=1e-3,  # prior_kernel_noise
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        minibatch_at_dataset_level: bool = False,  # if True, we can vectorize the loop ranging over the tasks
        dataset_batch_size: Optional[
            int
        ] = None,  # if None, and minibatch_at_dataset_level is True, then the datasets have to have the same size
        random_state: jax.random.PRNGKey = None,
    ):
        """
        :param input_dim: The dimensionality of input points
        :param output_dim: The dimensionality of output points. Only output_dim = 1 is currently supported
        :param domain: The domain on which to sample the points used for the marginal kl
        :param learning_mode: Can be one of "learn_mean", "learn_kernel" or "both"
        :param weight_decay: Weight decay on the parameters. Corresponds to setting a (certain)
            gaussian hyperprior on the parameters of the mean, kernel. We do not apply weight_decay
            on the likelihood parameters, since this coresponds to letting the noise go to softplus(0.0) = 0.69
       :param covar_module: Can be "NN", "SE" (Squared Exponential, i.e. RBF) or a Kernel object
        :param mean_module: Can be "NN", "constant", "zero", or a Mean object
        :param mean_nn_layers: Size specifications of the hidden (!) layers used in the mean feature maps
        :param kernel_nn_layers: Size specifications of the hidden (!) layers used in the kernel feature maps
        :param feature_dim: In case covar_module is NN, feature_dim denotes the dimensionality of the output
            of the MLP that is then fed through a RBF kernel
        :param task_batch_size: The number of tasks in a batch
        :param num_tasks: The number of tasks we intend to train on. Required for jax.jit reasons
        :param lr: The learning rate of the AdamW optimizer
        :param lr_decay: The learning rate decay. 1.0 means no decay
        :param prior_factor: How to weight the functional kl term, denoted as kappa in the paper
        :param train_data_in_kl: Whether to include the train data in the evaluation vector of the functional kl-divergence
            if yes, at most half the samples will come from the train data
        :param num_samples_kl: The number of samples at which we compute the kl (the dimensionality of the marginal
            distributions we compare)
        :param hyperprior_lengthscale: The lengthscale of the GP hyperprior in function space
        :param hyperprior_outputscale: The outputscale of the GP hyperprior in function space
        :param hyperprior_noise_std: The lengthscale of the GP hyperprior in function space
        :param normalize_data: Whether to do everything with normalized data
        :param normalizer: Optional normalizer object. If none supplied, normalization stats are inferred from the
            training data
        :param minibatch_at_dataset_level: Whether to draw minibatches per task, i.e. at the dataset level
        :param dataset_batch_size: If minibatch_at_dataset_level is true,
        :param random_state: A jax.random.PRNGKey to control all the pseudo-randomness inside this module
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

        assert isinstance(domain, ContinuousDomain) or isinstance(domain, DiscreteDomain)
        assert domain.d == input_dim, "Domain and input dimension don't match"
        assert learning_mode in [
            "learn_mean",
            "learn_kernel",
            "both",
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
            initial_noise_std=jax.nn.softplus(0.) + 0.001,
            kernel_length_scale=jax.nn.softplus(0.),
            kernel_output_scale=jax.nn.softplus(0.)
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
                variance=0.000001,  # does not matter, only used for prior
                learn_likelihood=False,
            )

            # the hyperprior is a dirac distribution over exactly one GP
            gp_hyperprior_particle = JAXExactGP(hyperprior_mean, hyperprior_covar, hyperprior_likelihood)
            return gp_hyperprior_particle.prior(xs, return_full_covariance=True)

        hyper_init, hyper_apply = hk.transform(hyperprior_impure)
        hyper_params = initialize_model(hyper_init, None, (1, input_dim))
        self.hyperprior_marginal = functools.partial(hyper_apply, hyper_params, None)
        self.hyperprior_noise_std = hyperprior_noise_std
        self._rng, sample_key = jax.random.split(self._rng)

        def target_post_prob_one_task(particle, key, xs, ys):
            mll, _ = self._apply_fns.base_learner_mll_estimator(particle, empty_state, None, xs, ys)
            kl = self._functional_kl(particle, key, xs)

            # this is formula (3) from pacoh, and prior_factor corresponds to kappa
            n = num_tasks
            m = xs.shape[0]

            return (
                -mll / (task_batch_size * m)
                + prior_factor * (1 / jnp.sqrt(n) + 1 / (n * m)) * kl / task_batch_size
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
            self.target_debugging_purposes = target_post_prob_vmap
        else:
            self.target_val_and_grad = jax.jit(jax.value_and_grad(target_post_prob_loop))
            self.target_debugging_purposes = target_post_prob_loop

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
        # self.target_debugging_purposes(self.particle, key, xs_batch, ys_batch)
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
            # do not use more than half of the training points for the functional kl
            n_train_x = min(
                xs_train.shape[0], self.num_samples_kl // 2
            )  # number of samples from inside the train set
            n_outside_x = self.num_samples_kl - n_train_x  # number of samples from domain
            k, indices_key = jax.random.split(k)
            idx_rand_train = jax.random.choice(
                k, xs_train.shape[0], (n_train_x,), replace=False
            )  # choose n indices for the train set,  np.random.choice(xs_train.shape[0], n_train_x)
            samples = self.domain_dist.sample(indices_key, (n_outside_x,))
            xs_kl = jnp.concatenate(
                [
                    xs_train[idx_rand_train],  # already normalized
                    self._normalizer.handle_data(samples)  # need normalization
                ],
                axis=0,
            )
        else:
            xs_kl = self._normalizer.handle_data(self.domain_dist.sample(k, (self.num_samples_kl,)))

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

        hyperposterior, _ = self._apply_fns.base_learner_predict(particle, self.empty_state, None, xs_kl, True)
        hyperprior = self.hyperprior_marginal(xs_kl)

        assert isinstance(
            hyperposterior, MultivariateNormal
        ), "both hyperprior and hyperposterior should have same shape"

        def compute_kl_with_noise(noise):
            # computes the kl divergence between hyperprior and hyperposterior, by adding a little diagonal noise to the
            # hyperprior
            diagonal_noise = noise * jnp.eye(xs_kl.shape[0])
            covar_with_noise = hyperprior.covariance_matrix + diagonal_noise
            hyperprior_with_noise = MultivariateNormal(hyperprior.mean, covar_with_noise)
            kl = multivariate_kl(hyperposterior, hyperprior_with_noise)
            return kl

        # this is both differentiable and jittable
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow
        # standard for loop is differentiable as condition does not depend on data
        # WARNING: except it's apparently not...

        inject_noise_std = self.hyperprior_noise_std
        kl = jnp.nan
        # kl = compute_kl_with_noise(inject_noise_std)
        for _ in range(5):
            kl = jax.lax.cond(
                jnp.isnan(kl),
                lambda n: compute_kl_with_noise(n),
                lambda _: kl,
                inject_noise_std,
            )
            inject_noise_std *= 2

        return kl

    def _clear_data(self):
        super()._clear_data()


if __name__ == "__main__":
    from pacoh.bo.meta_environment import RandomMixtureMetaEnv

    meta_env = RandomMixtureMetaEnv(random_state=np.random.RandomState(29))
    num_train_tasks = 20
    meta_train_data = meta_env.generate_uniform_meta_train_data(num_tasks=num_train_tasks, num_points_per_task=10)
    meta_test_data = meta_env.generate_uniform_meta_valid_data(num_tasks=50, num_points_context=10, num_points_test=160)

    NN_LAYERS = (32, 32)

    plot = True
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """
    print('\n ---- GPR mll meta-learning ---- ')

    prior_factor = 1e-2  # This is larger than in the original codebase by a factor of 10
    gp_model = F_PACOH_MAP_GP(1, 1, domain=meta_env.domain, random_state=jax.random.PRNGKey(3425),
                             num_tasks=num_train_tasks,
                             weight_decay=1e-4, prior_factor=prior_factor,
                             task_batch_size=2, covar_module='NN', mean_module='NN', lr=1e-3,
                             mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS)
    itrs = 0
    for i in range(10):
        def mapper(name, pname, val, length_scale, output_scale, likelihood_noise):
            if "likelihood" in name:
                return val*0.0+jnp.log(likelihood_noise)
            elif "LengthScale" in name:
                return val*0.0+jnp.log(length_scale)
            elif "OutputScale" in name:
                return val*0.0+jnp.log(output_scale)
            else:
                raise NotImplementedError("is problem")

        def newparticle(particle, length_scale, output_scale, likelihood_noise):
            return hk.data_structures.map(functools.partial(mapper, length_scale=length_scale, output_scale=output_scale, likelihood_noise=likelihood_noise), particle)

        # gp_model.particle = newparticle(gp_model.particle, 0.6931, 0.6931, 0.6941)

        gp_model.meta_fit(meta_train_data, meta_valid_tuples=meta_test_data, log_period=1000, num_iter_fit=200)
        itrs += 200


        task_xs, task_ys = gp_model.meta_train_tuples_debugging_purposes[0]

        x_plot = np.linspace(meta_env.domain.l, meta_env.domain.u, num=150)
        x_context, t_context, x_test, y_test = meta_test_data[0]
        pred_mean, pred_std = gp_model.meta_predict(x_context, t_context, x_plot, return_density=False)
        ucb, lcb = (pred_mean + 2 * pred_std).flatten(), (pred_mean - 2 * pred_std).flatten()

        plt.scatter(x_test, y_test)
        plt.scatter(x_context, t_context)

        print("AFTER ITERATION", itrs)
        print("first dataset", gp_model.meta_eval(x_context, t_context, x_test, y_test))
        print("all datasets", gp_model.eval_datasets(meta_test_data))
        print("params", jax.tree_map(jax.nn.softplus, gp_model.particle))




        plt.plot(x_plot, pred_mean)
        plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2)
        plt.title('GPR meta mll (prior_factor =  %.4f) itrs = %i' % (prior_factor, itrs))
        plt.show()

