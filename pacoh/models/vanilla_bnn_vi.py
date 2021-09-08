import functools
from typing import Callable

import jax as jax
from jax import vmap, numpy as jnp
import jax.nn
import tensorflow as tf
import haiku as hk
# from pacoh_nn.bnn.regression_algo import RegressionModel
# from pacoh_nn.modules.neural_network import BatchedFullyConnectedNN
# from pacoh_nn.modules.prior_posterior import GaussianPosterior, GaussianPrior
# from pacoh_nn.modules.likelihood import GaussianLikelihood
from pacoh.modules.abstract import RegressionModel


def get_batched_module(init_fn, apply_fns):
    """ Takes an init function and either a single apply funciton or a tuple thereof, and returns
        batched module versions of them. This means it initialises a number of models in paralels
        Args:
            init:  a haiku init function (key, data) -> params
            apply: a tuple or apply functions (params, key, data) -> Any
        Returns:
            init_batcbed: ([keys], data) -> [params]
            apply_batched: ([params], [keys], data) -> [Any]
            apply_batched_batched_inputs:  ([params], [keys], [data]) -> [Any]
        Notes:
            apply_fns is expected to be a named tuple, in which case this method returns named tuples of the same type
            init batched takes a number of PRNGKeys and one example input and returns that number of models
            apply_batched takes a number of parameter trees and same number of keys and one data batch and returns that number of inputs
            apply_batched_batched_inputs will assume that the data is batched on a per_model baseis, i.e. that every model gets
                a different data batch. This corresponds to batch_inputs=True in the call method in Jonas' code
       """

    """batched.init returns a batch of model parameters, which is why it takes n different random keys"""
    batched_init = vmap(init_fn, in_axes=(0,None))
    if callable(apply_fns):
        # there is only one apply function
        apply_batched = vmap(apply_fns, in_axes=(0,0,None))
        apply_batched_batched_inputs = vmap(apply_fns, in_axes=(0,0,0))
        return batched_init, apply_batched, apply_batched_batched_inputs
    else:
        apply_dict = {}
        apply_dict_batched_inputs = {}

        for fname, func in apply_fns._asdict().iteritems():
            apply_dict[fname] = vmap(func, in_axes=(0,0,None))
            apply_dict_batched_inputs[fname]  = vmap(func, in_axes=(0,0,0))

        return batched_init, apply_fns.__class__(**apply_dict), apply_fns.__class__(**apply_dict_batched_inputs)


def batched_module(constructor_fn):
    """ Decorator that takes a function returning pure functions and makes them batched """
    def batched_constructor_fn(*args, **kwargs):
        return get_batched_module(*constructor_fn(*args, **kwargs))

    return batched_constructor_fn


@batched_module
def get_pure_nn_functions(output_dim, hidden_layer_sizes, activation):
    def forward(xs):
        nn = hk.nets.MLP(output_sizes=hidden_layer_sizes+(output_dim,), activation=activation)
        return nn(xs)
    return hk.transform(forward)










# class BayesianNeuralNetworkVI(RegressionModel):
#
#     def __init__(self, x_train, y_train, hidden_layer_sizes=(32, 32), activation=jax.nn.elu,
#                  likelihood_std=0.1, learn_likelihood=True, prior_std=1.0, prior_weight=0.1,
#                  likelihood_prior_mean=tf.math.log(0.1), likelihood_prior_std=1.0,
#                  batch_size_vi=10, batch_size=8, lr=1e-3):
#
#         self.prior_weight = prior_weight
#         self.likelihood_std = likelihood_std
#         self.batch_size = batch_size
#         self.batch_size_vi = batch_size_vi
#
#         # data handling
#         self._process_train_data(x_train, y_train)
#
#         # setup nn
#         self.nn = BatchedFullyConnectedNN(self.batch_size_vi, self.output_dim, hidden_layer_sizes, activation)
#         self.nn.build((None, self.input_dim))
#
#         # setup prior
#         self.nn_param_size = self.nn.get_variables_stacked_per_model().shape[-1]
#         if learn_likelihood:
#             self.likelihood_param_size = self.output_dim
#         else:
#             self.likelihood_param_size = 0
#         self.prior = GaussianPrior(self.nn_param_size, nn_prior_std=prior_std,
#                                    likelihood_param_size=self.likelihood_param_size,
#                                    likelihood_prior_mean=likelihood_prior_mean,
#                                    likelihood_prior_std=likelihood_prior_std)
#
#         # Likelihood
#         self.likelihood = GaussianLikelihood(self.output_dim, self.batch_size_vi)
#
#         # setup posterior
#         self.posterior = GaussianPosterior(self.nn.get_variables_stacked_per_model(), self.likelihood_param_size)
#
#         # setup optimizer
#         self.optim = tf.keras.optimizers.Adam(lr)
#
#     def predict(self, x, num_posterior_samples=20):
#         # data handling
#         x = self._handle_input_data(x, convert_to_tensor=True)
#         x = self._normalize_data(x)
#
#         # nn prediction
#         y_pred_batches = []
#         likelihood_std_batches = []
#         for _ in range(num_posterior_samples // self.batch_size_vi):
#             sampled_params = self.posterior.sample((self.batch_size_vi,))
#             nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)
#             likelihood_std_batches.append(likelihood_std)
#             y_pred_batches.append(self.nn.call_parametrized(x, nn_params))
#         y_pred = tf.concat(y_pred_batches, axis=0)
#         likelihood_std = tf.concat(likelihood_std_batches, axis=0)
#
#         pred_dist = self.likelihood.get_pred_mixture_dist(y_pred, likelihood_std)
#
#         # unnormalize preds
#         y_pred = self._unnormalize_preds(y_pred)
#         pred_dist = self._unnormalize_predictive_dist(pred_dist)
#         return y_pred, pred_dist
#
#     @tf.function
#     def step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
#         with tf.GradientTape(watch_accessed_variables=False) as tape:
#             # keep in mind: len(trainable variables) = number of defined Variables in class and all parent classes
#             tape.watch(self.posterior.trainable_variables)
#
#             # sample batch of parameters from the posterior
#             sampled_params = self.posterior.sample((self.batch_size_vi,))
#             nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)
#
#             # compute log-likelihood
#             y_pred = self.nn.call_parametrized(x_batch, nn_params)  # (batch_size_vi, batch_size, 1)
#             avg_log_likelihood = self.likelihood.log_prob(y_pred, y_batch, likelihood_std)
#
#             # compute kl
#             kl_divergence = self.posterior.log_prob(sampled_params) - self.prior.log_prob(sampled_params)
#             avg_kl_divergence = tf.reduce_mean(kl_divergence) / self.num_train_samples
#
#             # compute elbo
#             elbo = - avg_log_likelihood + avg_kl_divergence * self.prior_weight
#
#         # compute gradient of elbo wrt posterior parameters
#         grads = tape.gradient(elbo, self.posterior.trainable_variables)
#         self.optim.apply_gradients(zip(grads, self.posterior.trainable_variables))
#         return elbo


if __name__ == '__main__':
    rds = jax.random.PRNGKey(42)
    batch_size_vi = 3
    init_batched, batched_forward, batched_forward_batch_inputs = get_pure_nn_functions(1, (32, 32), jax.nn.elu)
    print(init_batched)
    keys = jax.random.split(rds, batch_size_vi+1)
    rds = keys[0]
    init_keys = keys[1:]
    x = jnp.ones((1, 1), dtype=jnp.float32)

    all_params = init_batched(init_keys, init_keys)
    print(all_params)

    # import numpy as np
    #
    # np.random.seed(0)
    # tf.random.set_seed(0)
    #
    # d = 1  # dimensionality of the data
    #
    # n_train = 50
    # x_train = np.random.uniform(-4, 4, size=(n_train, d))
    # y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)
    #
    # n_val = 200
    # x_val = np.random.uniform(-4, 4, size=(n_val, d))
    # y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)
    #
    # nn = BayesianNeuralNetworkVI(x_train, y_train, hidden_layer_sizes=(32, 32), prior_weight=0.001, learn_likelihood=False)
    #
    # n_iter_fit = 2000
    # for i in range(10):
    #     nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
    #     if d == 1:
    #         x_plot = tf.range(-8, 8, 0.1)
    #         nn.plot_predictions(x_plot, iteration=(i + 1) * n_iter_fit, experiment="bnn_svgd", show=True)
