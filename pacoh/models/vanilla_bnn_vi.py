import functools
import warnings
from typing import Callable

import jax
import numpy as np
import optax
from jax import numpy as jnp

from pacoh.models.regression_base import RegressionModel
from pacoh.models.pure.pure_functions import construct_bnn_forward_fns
from pacoh.modules.belief import GaussianBelief, GaussianBeliefState
from pacoh.modules.distributions import get_mixture, VmappableIndependent
from pacoh.util.constants import (
    LIKELIHOOD_MODULE_NAME,
    MLP_MODULE_NAME,
    POSITIVE_PARAMETER_NAME,
)
from pacoh.util.initialization import initialize_batched_model
from pacoh.util.tree import Tree
from pacoh.util.data_handling import DataNormalizer, normalize_predict


def neg_elbo(
    posterior: GaussianBeliefState,
    key: jax.random.PRNGKey,
    x_batch: jnp.array,
    y_batch: jnp.array,
    prior: GaussianBeliefState,
    prior_weight: float,
    batch_size_vi: int,
    num_train_points: int,
    log_prob_fn: Callable[[Tree, jax.random.PRNGKey, jnp.array, jnp.array], jnp.array],
    batch_pred_fn: Callable[[Tree, jax.random.PRNGKey, jnp.array], jnp.array],
):
    """
    :param posterior: dict of GaussianBeliefState
    :param key: jax.random.PRNGkey
    :param x_batch: data
    :param y_batch:data
    :param prior: dict of GaussianBeliefState
    :param prior_weight: how heavy to weight the kl term between prior and posterior
    :param batch_size_vi: how many samples to use for approximating the elbo expectation
    :param num_train_samples: the total number of train_samples
    :param nn_apply: the pure nn forward function
    :param likelihood_applys: the pure likelihood functions
    :param learn_likelihood: whether to use
    :param fixed_likelihood_params: should be None if learn_likelihood and a tree of parameters otherwise
    :return elbo
    Notes:
        x_batch and y_batch should live in the normalized space, which is the case when they are the stored data
    """
    # sample from posterior and predict
    batched_params = GaussianBelief.rsample(posterior, key, batch_size_vi)
    ys_pred = batch_pred_fn(batched_params, None, x_batch)

    # compute log likelihood of the data
    ys_true_rep = jnp.repeat(jnp.expand_dims(y_batch, axis=0), batch_size_vi, axis=0)
    log_likelihoods = log_prob_fn(batched_params, None, ys_true_rep, ys_pred)
    avg_log_likelihood = jnp.mean(log_likelihoods)

    # kl computation (regularization)
    kl_divergence = GaussianBelief.log_prob(posterior, batched_params) - GaussianBelief.log_prob(
        prior, batched_params
    )
    avg_kl_divergence = jnp.mean(kl_divergence)

    # result computation
    return -avg_log_likelihood + (prior_weight / num_train_points) * avg_kl_divergence


class BayesianNeuralNetworkVI(RegressionModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        random_state: jax.random.PRNGKey = None,
        hidden_layer_sizes=(32, 32),
        activation: Callable = jax.nn.elu,
        learn_likelihood: bool = True,
        prior_std: float = 0.1,
        prior_weight: float = 0.1,
        likelihood_prior_mean: float = 0.1,
        likelihood_prior_std: float = 0.0,
        batch_size_vi: int = 10,
        batch_size: int = 8,
        lr: float = 1e-2,
    ):
        """
        :param input_dim: The dimension of a data point
        :param normalize_data: Whether to normalize the data
        :param random_state: A top level jax.PRNGKey
        :param hidden_layer_sizes: The hidden layers of the BNN
        :param activation: The activation function of the BNN
        :param likelihood_std: The mean of the likelihood std parameter
        :param learn_likelihood: Whether to interpret the likelihood std as a learnable variable with a mean and std itself
        :param prior_std: The std of the prior on each NN weight
        :param prior_weight: Multiplicative factor before the KL divergence term in the ELBO
        :param likelihood_prior_mean:
        :param likelihood_prior_std: The sigma in sigmall sim N(0,sigma)
        :param batch_size_vi: The number of samples from the posterior to approximate the expectation in the ELBO
        :param batch_size: The number of data points you get while training and predicting
        :param lr: The learning rate to use with the ELBO gradients
        """
        super().__init__(
            input_dim,
            output_dim,
            normalize_data,
            normalizer,
            flatten_ys=False,
            random_state=random_state,
        )
        self._batch_size = batch_size
        self.batch_size_vi = batch_size_vi

        # a) Get batched forward functions for the nn and likelihood
        self._rng, init_key = jax.random.split(self._rng)
        init, self.apply, self.apply_broadcast = construct_bnn_forward_fns(
            output_dim,
            hidden_layer_sizes,
            activation,
            likelihood_prior_mean,
            learn_likelihood,
        )

        # b) Initialize the prior and posterior
        params, template = initialize_batched_model(init, batch_size_vi, init_key, (batch_size, input_dim))

        def mean_std_map(mod_name: str, name: str, __: jnp.array):
            transform = lambda value, name: np.log(value) if POSITIVE_PARAMETER_NAME in name else value
            if LIKELIHOOD_MODULE_NAME in mod_name:
                return transform(likelihood_prior_mean, name), likelihood_prior_std
            elif MLP_MODULE_NAME in mod_name:
                return 0.0, prior_std
            else:
                raise AssertionError("Unknown hk.Module: can only handle mlp and likelihood")

        self.posterior = self.prior = GaussianBeliefState.initialize_heterogenous(mean_std_map, template)

        # c) Setup optimizer: posterior parameters
        lr_scheduler = optax.constant_schedule(lr)
        self.optimizer = optax.adam(lr_scheduler)
        self.optimizer_state = self.optimizer.init(self.posterior)

        # d) Setup pure objective function and its derivative
        elbo_fn = functools.partial(
            neg_elbo,
            prior=self.prior,
            prior_weight=prior_weight,
            batch_size_vi=self.batch_size_vi,
            log_prob_fn=self.apply_broadcast.log_prob,
            batch_pred_fn=self.apply.pred,
        )

        self.elbo = elbo_fn
        self.elbo_val_and_grad = jax.jit(jax.value_and_grad(jax.jit(elbo_fn)))
        self.pred_dist_apply = jax.jit(self.apply.pred_dist)

    def _recompute_posterior(self, num_iter_fit=500):
        """Fits the underlying GP to the currently stored datapoints."""
        warnings.warn("_recompute posterior does nothing for now")

    def fit(self, xs_val=None, ys_val=None, log_period=500, num_iter_fit=None):
        super().fit(xs_val, ys_val, log_period, num_iter_fit)

    @normalize_predict
    def predict(self, xs, num_posterior_samples=1):
        self._rng, key = jax.random.split(self._rng)
        nn_params = GaussianBelief.rsample(self.posterior, key, num_posterior_samples)
        preds = self.pred_dist_apply(nn_params, None, xs)
        ret = get_mixture(preds, num_posterior_samples)
        return ret

    def _step(self, x_batch, y_batch):
        self._rng, step_key = jax.random.split(self._rng)
        nelbo, gradelbo = self.elbo_val_and_grad(
            self.posterior,
            step_key,
            x_batch,
            y_batch,
            num_train_points=self._num_train_points,
        )
        updates, new_opt_state = self.optimizer.update(gradelbo, self.optimizer_state, self.posterior)
        self.optimizer_state = new_opt_state
        self.posterior = optax.apply_updates(self.posterior, updates)
        return nelbo


if __name__ == "__main__":
    pass
    np.random.seed(1)
    from jax.config import config

    config.update("jax_debug_nans", False)
    config.update("jax_disable_jit", False)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200

    x_plot = np.linspace(-8, 8, num=n_val)
    x_plot = np.expand_dims(x_plot, -1)
    y_val = np.sin(x_plot) + np.random.normal(scale=0.1, size=x_plot.shape)

    nn = BayesianNeuralNetworkVI(
        input_dim=d,
        output_dim=1,
        batch_size_vi=1,
        hidden_layer_sizes=(32, 32),
        prior_weight=0.0,
        learn_likelihood=False,
    )
    nn.add_data_points(x_train, y_train)

    n_iter_fit = 200  # 2000
    for i in range(200):
        nn.fit(log_period=100, num_iter_fit=n_iter_fit, xs_val=x_plot, ys_val=y_val)
        from matplotlib import pyplot as plt

        pred = nn.predict(x_plot)
        lcb, ucb = nn.confidence_intervals(x_plot)
        plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), alpha=0.3)
        plt.plot(x_plot, pred.mean)
        plt.scatter(x_train, y_train)

        plt.show()

    # https://github.com/tensorflow/probability/issues/1271

    # def get_dist(mean):
    #     return VmappableIndependent(numpyro.distributions.Normal(mean, jnp.ones_like(mean)), mean.ndim)
    #
    # get_dists = jax.vmap(get_dist)
    #
    # key = jax.random.PRNGKey(42)
    # mean = jax.random.normal(key, (8, 1))
    # means = jax.random.normal(key, (3, 8, 1))
    #
    # simple_dist = get_dist(mean)
    # batched_dist = get_dists(means)
    #
    # print(simple_dist.batch_shape, "&", simple_dist.event_shape)  # Prints: () & (8,1)
    # print(batched_dist.batch_shape, "&", batched_dist.event_shape)  # Prints: (3,) & (8,1)
