import functools
import warnings
from typing import NamedTuple, Any

import jax
import tensorflow as tf
import haiku as hk

from jax import numpy as jnp

from pacoh.modules.abstract import RegressionModel
from pacoh.modules.util import pytrees_unstack, _handle_batch_input_dimensionality
from pacoh.modules.bnn.batched_modules import transform_and_batch_module, multi_transform_and_batch_module
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.modules.priors_posteriors import GaussianPosterior, GaussianPrior

@transform_and_batch_module
def get_pure_batched_nn_functions(output_dim, hidden_layer_sizes, activation):
    def nn_forward(xs):
        nn = hk.nets.MLP(output_sizes=hidden_layer_sizes+(output_dim,), activation=activation)
        return nn(xs)

    return nn_forward

class LikelihoodInterface(NamedTuple):
    log_prob: Any
    add_noise: Any

@functools.partial(multi_transform_and_batch_module, num_data_args={'log_prob': 2, 'add_noise': 1})
def get_pure_batched_likelihood_functions():

    def factory():
        likelihood = JAXGaussianLikelihood() # TODO: non-constant variance over dimensions

        def log_prob(xs, ys):
            return likelihood.log_prob(xs, ys)

        def add_noise(pred_f):
            return likelihood(pred_f)

        return add_noise, LikelihoodInterface(log_prob=log_prob, add_noise=add_noise)

    return factory

def get_prior_function_like(pytree):
    # returns a prior or posterior
    pass



class BayesianNeuralNetworkVI(RegressionModel):

    def __init__(self, input_dim: int, output_dim: int, normalize_data: bool = True, random_state: jax.random.PRNGKey = None,
                 hidden_layer_sizes=(32,32), activation=jax.nn.elu,
                 likelihood_std=0.1, learn_likelihood=True, prior_std=1.0, prior_weight=0.1,
                 likelihood_prior_mean=tf.math.log(0.1), likelihood_prior_std=1.0,
                 batch_size_vi=10, batch_size=8, lr=1e-3):
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
        :param likelihood_prior_mean: TODO figure out why this is not the same thing as the likelihood std
        :param likelihood_prior_std: The sigma in sigmall sim N(0,sigma)
        :param batch_size_vi: The number of samples from the posterior to approximate the expectation in the ELBO
        :param batch_size: The number of data points you get while training and predicting
        :param lr: The learning rate to use with the ELBO gradients
        """
        super().__init__(input_dim, output_dim, normalize_data, random_state)

        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size
        self.batch_size_vi = batch_size_vi

        """ A) Setup neural network module """
        # TODO this should be just a batched nn forward function
        keys = jax.random.split(self._rds, self.batch_size_vi + 1)
        self._rds = keys[0]
        init_keys = keys[1:]
        batched_nn_init_fn, self.batched_nn_apply_fn, _ = get_pure_batched_nn_functions(output_dim,
                                                                                             hidden_layer_sizes,
                                                                                             activation)

        # note that some of this is not strictly necessary => we don't really need to initialize parameters, as we will
        # be sampling them anyways. TODO think about if we can ditch this alltoghether
        # TODO is the hk.nets.MLP already supporting batched xs?

        dummy_input = jnp.zeros((self.batch_size_vi, self.batch_size, input_dim))
        dummy_params = batched_nn_init_fn(init_keys, dummy_input)
        dummy_single_param = pytrees_unstack(dummy_params)[0] # just care about the form of the first models params

        """ B) Setup likelihood module """
        batched_likelihood_init_fn, self.batched_likelihood_apply_fn = get_pure_batched_likelihood_functions()

        # TODO if likelihood is not learned
        likelihood_input = jnp.zeros((self.batch_size_vi, self.batch_size, output_dim))
        likelihood_params = batched_likelihood_init_fn(init_keys, likelihood_input)
        likelihood_single_param = pytrees_unstack(likelihood_params)[0]


        """ C) Setup prior module """
        # If the last 3 arguments are none, the prior is not learnable
        if learn_likelihood:
            prior_init, self.prior_sample = get_sampling_function_like(nn_params_like=dummy_single_param,
                                                                           nn_params_std=prior_std,
                                                                           lh_params_like=likelihood_single_param,
                                                                           lh_prior_mean=likelihood_prior_mean,
                                                                           lh_prior_std=likelihood_prior_std)
        else:
            prior_init, self.prior_sample = get_sampling_function_like(nn_params_like=dummy_single_param,
                                                                            nn_params_std=prior_std)

        # The prior parameters shouldn't change over time, so we can directly apply the params (first arguments)
        # for convenience
        _prior_params = prior_init(init_keys) # I think no data is needed for init
        self.prior_sample = functools.partial(self.prior_sample, _prior_params)


        """ Setup posterior module. """
        posterior_init, self.posterior_sample = get_sampling_function_like(nn_params_like=dummy_single_param,
                                                                           lh_params_like=likelihood_single_param)


        self._rds, init_keys = get_prng_keys(self._rds, 1)
        self.posterior_params = posterior_init(init_keys)


        # setup optimizer
        warnings.warn("setup optimizer")
        self.optim = tf.keras.optimizers.Adam(lr)


    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints. """
        self._rds, fit_key = jax.random.split(self._rds)
        _, self._state = self._apply_fns.fit_fn(self._params, self._state, fit_key, self.xs_data, self.ys_data)


    def predict(self, xs, num_posterior_samples=20):
        # data handling
        xs = _handle_batch_input_dimensionality(xs)
        xs = self._normalize_data(xs)

        # nn prediction
        y_pred_batches = []
        likelihood_std_batches = []

        # TODO check that I really don't need a for loop
        sampled_params = self.posterior_sample((num_posterior_samples,))
        sampled_nn_params, sampled_stds = pytree_split(sampled_params)
        ys_pred = self.batched_nn_apply_fn(sampled_nn_params, xs)
        pred_dist = get_pred_mixture_dist(ys_pred, sampled_stds) # TODO implement
        ys_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return ys_pred, pred_dist

        # for _ in range(num_posterior_samples // self.batch_size_vi):
        #
        #     sampled_params = self.posterior.sample((self.batch_size_vi,))
        #     nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)
        #     likelihood_std_batches.append(likelihood_std)
        #     y_pred_batches.append(self.nn.call_parametrized(x, nn_params))
        # y_pred = tf.concat(y_pred_batches, axis=0)
        # likelihood_std = tf.concat(likelihood_std_batches, axis=0)

        # pred_dist = self.likelihood.get_pred_mixture_dist(y_pred, likelihood_std)

        # unnormalize preds


    @tf.function
    def step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # keep in mind: len(trainable variables) = number of defined Variables in class and all parent classes
            tape.watch(self.posterior.trainable_variables)

            # sample batch of parameters from the posterior
            sampled_params = self.posterior.sample((self.batch_size_vi,))
            nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)

            # compute log-likelihood
            y_pred = self.nn.call_parametrized(x_batch, nn_params)  # (batch_size_vi, batch_size, 1)
            avg_log_likelihood = self.likelihood.log_prob(y_pred, y_batch, likelihood_std)

            # compute kl
            kl_divergence = self.posterior.log_prob(sampled_params) - self.prior.log_prob(sampled_params)
            avg_kl_divergence = tf.reduce_mean(kl_divergence) / self.num_train_samples

            # compute elbo
            elbo = - avg_log_likelihood + avg_kl_divergence * self.prior_weight

        # compute gradient of elbo wrt posterior parameters
        grads = tape.gradient(elbo, self.posterior.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.posterior.trainable_variables))
        return elbo


if __name__ == '__main__':
    import numpy as np

    np.random.seed(0)
    tf.random.set_seed(0)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200
    x_val = np.random.uniform(-4, 4, size=(n_val, d))
    y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)

    nn = BayesianNeuralNetworkVI(x_train, y_train, hidden_layer_sizes=(32, 32), prior_weight=0.001, learn_likelihood=False)

    n_iter_fit = 2000
    for i in range(10):
        nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
        if d == 1:
            x_plot = tf.range(-8, 8, 0.1)
            nn.plot_predictions(x_plot, iteration=(i + 1) * n_iter_fit, experiment="bnn_svgd", show=True)