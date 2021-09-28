import functools
import warnings
from typing import NamedTuple, Any

import jax
import numpyro.distributions
import optax
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
    get_posterior_from_means: Any

@functools.partial(multi_transform_and_batch_module, num_data_args={'log_prob': 2, 'add_noise': 1})
def get_pure_batched_likelihood_functions():

    def factory():
        likelihood = JAXGaussianLikelihood() # TODO: non-constant variance over dimensions

        def log_prob(xs, ys):
            return likelihood.log_prob(xs, ys)

        def add_noise(pred_f): # add noise to a predictive distribution
            return likelihood(pred_f)

        def get_posterior_from_means(ys_pred): # add noise to a mean prediction (same as add_noise with a zero_variance_pred_f)
            return likelihood.get_posterior_from_means(ys_pred)

        return add_noise, LikelihoodInterface(log_prob=log_prob, add_noise=add_noise, get_posterior_from_means=get_posterior_from_means)

    return factory


def sample_gaussian_tree(mean_tree, std_tree, key, n_samples):
    num_params = len(jax.tree_util.leaves(mean_tree)) # number of prng keys we need
    key, *keys = jax.random.split(key, num_params+1) # generate some key leaves
    keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mean_tree), keys) # build tree structure with keys

    def sample_leaf(mean, std, key):
        mean = jnp.expand_dims(mean, -1)
        std = jnp.expand_dims(std, -1)

        return mean + jax.random.normal(key, (n_samples, *mean.shape), dtype=jnp.float32) * std

    return jax.tree_multimap(sample_leaf, mean_tree, std_tree, keys_tree)


def broadcast_params(tree):
    return jax.tree_map(functools.partial(jnp.repeat, axis=0), tree)


def logprob(posterior_params, sampled_nn_params, sampled_lh_params):
    raise NotImplementedError


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
        dummy_input = jnp.zeros((self.batch_size_vi, self.batch_size, input_dim))
        dummy_params = batched_nn_init_fn(init_keys, dummy_input)
        dummy_single_param = pytrees_unstack(dummy_params)[0] # just care about the form of the first models params

        """ B) Setup likelihood module """
        batched_likelihood_init_fn, self.batched_likelihood_apply_fns = get_pure_batched_likelihood_functions()
        likelihood_input = jnp.zeros((self.batch_size_vi, self.batch_size, output_dim))
        likelihood_params = batched_likelihood_init_fn(init_keys, likelihood_input)
        likelihood_single_param = pytrees_unstack(likelihood_params)[0]


        """ C) Setup prior and posterior module """
        # If the last 3 arguments are none, the prior is not learnable
        nn_prior_std = jax.tree_map(lambda p: jnp.ones(p.shape)*prior_std, dummy_single_param)
        nn_prior_mean = jax.tree_map(lambda p: jnp.zeros(p.shape), dummy_single_param)
        lh_prior_std = jax.tree_map(lambda p: jnp.ones(p.shape) * likelihood_prior_std, likelihood_single_param)
        lh_prior_mean = jax.tree_map(lambda p: jnp.ones(p.shape) * likelihood_prior_mean, likelihood_single_param)

        self.prior_params = {
            'nn_mean': nn_prior_std,
            'nn_std': nn_prior_mean,
            'lh_mean': lh_prior_mean,
            'lh_std': lh_prior_std
        }

        self.posterior_params = jax.tree_map(lambda v: v.copy(), self.prior_params)

        # TODO understand why Jonas initializes the prior differently for learnable and non-learnable likelihood?
        # TODO logscale
        """ D) Setup Posterior Module """
        self.learn_likelihood = learn_likelihood
        if not learn_likelihood:
            self.posterior_params['lh_std'] = None
            self.prior_params['lh_std'] = None

        """ E) Setup optimizer: posterior parameters """
        # TODO mask weight decay for log_scale parameters
        lr_scheduler = optax.constant_schedule(lr)
        self.optimizer = optax.adam(lr_scheduler)
        self.optimizer_state = self.optimizer.init(self.posterior_params)


    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints. """
        # TODO ask Jonas what about recomputing the posterior -> start from scratch?  Not in an online setting I suppose
        raise NotImplementedError


    def predict(self, xs, num_posterior_samples=20):
        # a) data handling
        xs = _handle_batch_input_dimensionality(xs)
        xs = self._normalize_data(xs)

        # b) nn prediction
        self._rds, nn_key, lh_key = jax.random.split(self._rds, 3)

        nn_params = sample_gaussian_tree(self.posterior_params['nn_mean'], self.posterior_params['nn_std'], nn_key, num_posterior_samples)
        ys_pred = self.batched_nn_apply_fn(nn_params, None, xs)

        # c) get posterior
        if self.learn_likelihood:
            lh_params = sample_gaussian_tree(self.posterior_params['lh_mean'], self.posterior_params['lh_std'], lh_key, num_posterior_samples)
        else:
            lh_params = broadcast_params(self.posterior_params['lh_mean'])
        pred_dists = self.batched_likelihood_apply_fns.get_posterior_from_means(lh_params, ys_pred)
        mixture = numpyro.distributions.Categorical(probs=jnp.ones((self.batch_size_vi,))/self.batch_size_vi)
        pred_mixture = numpyro.distributions.MixtureSameFamily(mixture, pred_dists)
        # TODO affinetransformed distribution stuff

        # d) data handling undo
        ys_pred = self._unnormalize_preds(ys_pred) # should that be some kind of mean??
        pred_dist = self._unnormalize_predictive_dist(pred_mixture)
        return ys_pred, pred_dist

    def _step(self, x_batch, y_batch):
        def negelbo(posterior_params, x_batch, y_batch):
            # sample from posterior
            sampled_nn_params = sample_gaussian_tree(posterior_params['nn_mean'], posterior_params['nn_std'])
            if self.learn_likelihood:
                sampled_lh_params = sample_gaussian_tree(posterior_params['lh_mean'], posterior_params['lh_std'])
            else:
                sampled_lh_params = broadcast_params(posterior_params['lh_mean'])

            # predict and compute log-likelihood
            ys_pred = self.batched_nn_apply_fn(sampled_nn_params, None, x_batch)
            avg_log_likelihood = self.batched_likelihood_apply_fns.log_prob(sampled_lh_params, y_batch, ys_pred)

            # compute kl
            kl_divergence = logprob(posterior_params, sampled_nn_params, sampled_lh_params) \
                          - logprob(self.prior_params, sampled_nn_params, sampled_lh_params)

            avg_kl_divergence = jnp.reduce_mean(kl_divergence) / self.num_train_samples

            # compute elbo
            negelbo = - avg_log_likelihood + avg_kl_divergence * self.prior_weight
            return negelbo

        nelbo, gradelbo = jax.value_and_grad(negelbo)(self.posterior_params, x_batch, y_batch)
        updates, new_opt_state = self.optimizer.update(gradelbo, self.optimizer_state, self.posterior_params)
        self.posterior_params = optax.apply_updates(self.posterior_params, updates)

        return nelbo



if __name__ == '__main__':
    import haiku as hk
    from jax import numpy as jnp
    import jax
    class MyMod(hk.Module):
        def fwd(self):
            return hk.get_parameter("hi", [3], init=hk.initializers.Constant(1.2))

    def impure():
        return MyMod().fwd()

    init, apply = hk.transform(impure)
    param = init(None)

    flat, structure = jax.tree_flatten(param)
    flat = flat[0]
    print(flat)
    n_samples = 2
    flatsamps = [jax.random.normal(jax.random.PRNGKey(1), (flat.size, n_samples)) + jnp.expand_dims(flat, -1)]
    sampled_params = jax.tree_util.tree_unflatten(structure, flatsamps)

    print(structure)
    print("")
    print(sampled_params)
#     import numpy as np
#
#     np.random.seed(0)
#     tf.random.set_seed(0)
#
#     d = 1  # dimensionality of the data
#
#     n_train = 50
#     x_train = np.random.uniform(-4, 4, size=(n_train, d))
#     y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)
#
#     n_val = 200
#     x_val = np.random.uniform(-4, 4, size=(n_val, d))
#     y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)
#
#     nn = BayesianNeuralNetworkVI(x_train, y_train, hidden_layer_sizes=(32, 32), prior_weight=0.001, learn_likelihood=False)
#
#     n_iter_fit = 2000
#     for i in range(10):
#         nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
#         if d == 1:
#             x_plot = tf.range(-8, 8, 0.1)
#             nn.plot_predictions(x_plot, iteration=(i + 1) * n_iter_fit, experiment="bnn_svgd", show=True)
#
#
# i