import functools
from typing import Callable, Dict

import jax
import numpy as np
import optax
from jax import numpy as jnp
from tqdm import trange

from pacoh.models.regression_base import RegressionModel
from pacoh.modules.distributions import AffineTransformedDistribution, get_mixture
from pacoh.modules.pure_functions import get_pure_batched_likelihood_functions, \
    get_pure_batched_nn_functions
from pacoh.modules.pure_interfaces import LikelihoodInterface
from pacoh.modules.priors_posteriors import GaussianBelief, GaussianBeliefState
from pacoh.util.initialization import initialize_batched_model
from pacoh.util.tree import pytree_unstack, Tree, broadcast_params
from pacoh.util.data_handling import handle_batch_input_dimensionality

def neg_elbo(posterior: Dict[str, GaussianBeliefState],
             key: jax.random.PRNGKey,
             x_batch: jnp.array,
             y_batch: jnp.array,
             prior: Dict[str, GaussianBeliefState],
             prior_weight: float,
             batch_size_vi: int,
             num_train_points: int,
             nn_apply: Callable[[Tree, jax.random.PRNGKey, jnp.array], jnp.array],
             likelihood_applys: LikelihoodInterface,
             learn_likelihood: bool,
             fixed_likelihood_params: Tree = None):
    """
    :param posterior: dict of GaussianBeliefState
    :param key: jax.random.PRNGkey
    :param x_batch: data
    :param y_batch:data
    :param prior: dict of GaussianBeliefState
    :param prior_weight: how heavy to weight the kl term between prior and posterior
    :param batch_size_vi: how many samples to use for approximating the elbo expectation
    :param num_train_samples: the total number of train_samples (TODO this could be handled via the prior weight part)
    :param nn_apply: the pure nn forward function
    :param likelihood_applys: the pure likelihood functions
    :param learn_likelihood: whether to use
    :param fixed_likelihood_params: should be None if learn_likelihood and a tree of parameters otherwise
    :return elbo
    """
    nn_key, lh_key = jax.random.split(key)

    # sample from posterior
    nn_params_batch = GaussianBelief.rsample(posterior['nn'], nn_key, batch_size_vi)
    if learn_likelihood:
        lh_params_batch = GaussianBelief.rsample(posterior['lh'], lh_key, batch_size_vi)
    else:
        lh_params_batch = fixed_likelihood_params

    # predict
    ys_pred = nn_apply(nn_params_batch, None, x_batch)
    ys_true_rep = jnp.repeat(jnp.expand_dims(y_batch, axis=0), batch_size_vi, axis=0)

    # data log likelihood
    log_likelihoods = likelihood_applys.log_prob(lh_params_batch, None, ys_true_rep, ys_pred) # (batch_size_vi, batch_size, output_dimension)
    avg_log_likelihood = jnp.mean(log_likelihoods)  # this is the average log likelihood on

    # kl computation (regularization)
    kl_divergence = GaussianBelief.log_prob(posterior['nn'], nn_params_batch) \
                    - GaussianBelief.log_prob(prior['nn'], nn_params_batch)  # (batch_size_vi,)

    avg_kl_divergence = jnp.mean(kl_divergence)

    if learn_likelihood:
        kl_divergence_lh = GaussianBelief.log_prob(posterior['lh'], lh_params_batch) \
                            - GaussianBelief.log_prob(prior['lh'], lh_params_batch)

        avg_kl_divergence += jnp.mean(kl_divergence_lh)

    # result computation
    return -avg_log_likelihood + (prior_weight / num_train_points) * avg_kl_divergence



class BayesianNeuralNetworkVI(RegressionModel):
    def __init__(self, input_dim: int, output_dim: int, normalize_data: bool = True,
                 normalization_stats: Dict[str, np.array] = None,
                 random_state: jax.random.PRNGKey = None,
                 hidden_layer_sizes=(32, 32), activation=jax.nn.elu,
                 likelihood_std=0.1, learn_likelihood=True, prior_std=1.0, prior_weight=0.1,
                 likelihood_prior_mean=jnp.log(0.1), likelihood_prior_std=1.0,
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
        :param likelihood_prior_mean:
        :param likelihood_prior_std: The sigma in sigmall sim N(0,sigma)
        :param batch_size_vi: The number of samples from the posterior to approximate the expectation in the ELBO
        :param batch_size: The number of data points you get while training and predicting
        :param lr: The learning rate to use with the ELBO gradients
        """
        super().__init__(input_dim, normalize_data, normalization_stats, random_state)
        self.output_dim = output_dim
        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size
        self.batch_size_vi = batch_size_vi
        self._set_normalization_stats(normalization_stats)

        """ A) Get batched forward functions for the nn and likelihood"""
        keys = jax.random.split(self._rng, self.batch_size_vi + 1)
        self._rng = keys[0]
        init_keys = keys[1:]
        batched_nn_init_fn, self.batched_nn_apply_fn, _ = get_pure_batched_nn_functions(output_dim,
                                                                                        hidden_layer_sizes,
                                                                                        activation)

        batched_likelihood_init_fn, self.batched_likelihood_apply_fns, self.batched_likelihood_apply_fns_batch_inputs = get_pure_batched_likelihood_functions(likelihood_prior_mean)

        """ B) Initialize the parameters of the nns and likelihoods """
        nn_params, nn_param_template = initialize_batched_model(batched_nn_init_fn, nn_key, (self.batch_size, input_dim))
        likelihood_params, likelihood_params_template = initialize_batched_model(batched_likelihood_init_fn, lh_key, (self.batch_size, output_dim))

        """ C) Setup prior and posterior modules """
        nn_prior_state = GaussianBeliefState.initialize(0.0, prior_std, nn_param_template)

        self.prior = {
            'nn': nn_prior_state,
        }
        self.posterior = {
            'nn': nn_prior_state,
        }

        self.learn_likelihood = learn_likelihood
        if learn_likelihood:
            lh_prior_state = GaussianBeliefState.initialize(likelihood_prior_mean, likelihood_prior_std, likelihood_param_template)
            self.prior['lh'] = lh_prior_state
            self.posterior['lh'] = lh_prior_state
            self.fixed_likelihood_params = None
        else:
            self.fixed_likelihood_params = likelihood_params

        """ D) Setup optimizer: posterior parameters """
        lr_scheduler = optax.constant_schedule(lr)
        self.optimizer = optax.adam(lr_scheduler)
        self.optimizer_state = self.optimizer.init(self.posterior)

        """ E) Setup pure objective function """
        elbo_fn = functools.partial(neg_elbo,
                                    prior=self.prior,
                                    prior_weight=self.prior_weight,
                                    batch_size_vi=self.batch_size_vi,
                                    nn_apply=self.batched_nn_apply_fn,
                                    likelihood_applys=self.batched_likelihood_apply_fns_batch_inputs,
                                    learn_likelihood=self.learn_likelihood,
                                    fixed_likelihood_params=self.fixed_likelihood_params)

        self.elbo_fn = elbo_fn
        self.elbo_val_and_grad = jax.jit(jax.value_and_grad(elbo_fn))
        self.batched_nn = jax.jit(self.batched_nn_apply_fn)

    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints. """
        # TODO ask Jonas what about recomputing the posterior -> start from scratch?  Not in an online setting I suppose
        pass

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        train_batch_sampler = self._get_batch_sampler(self.xs_data, self.ys_data, self.batch_size)
        loss_list = []
        pbar = trange(num_iter_fit)
        for i in pbar:
            x_batch, y_batch = next(train_batch_sampler)
            loss = self._step(x_batch, jnp.expand_dims(y_batch, axis=-1))
            loss_list.append(loss)

            if i % log_period == 0:
                loss = jnp.mean(jnp.array(loss_list))
                loss_list = []
                message = dict(loss=loss)
                # if x_val is not None and y_val is not None:
                #     metric_dict = self.eval(x_val, y_val)
                #     message.update(metric_dict)
                pbar.set_postfix(message)

    def predict(self, xs, num_posterior_samples=20, return_density=True):
        # a) data handling
        xs = handle_batch_input_dimensionality(xs)
        xs = self._normalize_data(xs)

        # b) nn prediction
        self._rng, nn_key, lh_key = jax.random.split(self._rng, 3)

        nn_params = GaussianBelief.rsample(self.posterior['nn'], nn_key, num_posterior_samples)
        ys_pred = self.batched_nn(nn_params, None, xs)

        if self.learn_likelihood:
            lh_params = GaussianBelief.rsample(self.posterior['lh'], lh_key, num_posterior_samples)
        else:
            lh_params = broadcast_params(pytree_unstack(self.fixed_likelihood_params, 1)[0], num_posterior_samples)

        pred_dists = self.batched_likelihood_apply_fns_batch_inputs.get_posterior_from_means(lh_params, None, ys_pred)
        pred_mixture = get_mixture(pred_dists, self.batch_size_vi)
        # # shape is (num_posterior_samples, batch_size, output_dim) -> transformed to (batch_size, output_dim, num_posterior_samples)
        # pred_dists = stack_distributions(pred_dists)
        # 
        # 
        # mixture = numpyro.distributions.Categorical(probs=jnp.ones((num_posterior_samples,)) / num_posterior_samples)
        # 
        # pred_mixture = numpyro.distributions.MixtureSameFamily(mixture, pred_dists)

        pred_mixture_transformed = AffineTransformedDistribution(pred_mixture,
                                                                 normalization_mean=self.y_mean,
                                                                 normalization_std=self.y_std)
        if return_density:
            return pred_mixture_transformed
        else:
            return pred_mixture_transformed.mean, pred_mixture_transformed.stddev

    def _step(self, x_batch, y_batch):
        self._rng, step_key = jax.random.split(self._rng)
        nelbo, gradelbo = self.elbo_val_and_grad(self.posterior, step_key, x_batch, y_batch, num_train_points=self._num_train_points)
        updates, new_opt_state = self.optimizer.update(gradelbo, self.optimizer_state, self.posterior)
        self.optimizer_state = new_opt_state
        self.posterior = optax.apply_updates(self.posterior, updates)
        return nelbo

if __name__ == '__main__':
    np.random.seed(1)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200

    x_plot = np.linspace(-8, 8, num=n_val)
    x_plot = np.expand_dims(x_plot, -1)
    y_val = np.sin(x_plot) + np.random.normal(scale=0.1, size=x_plot.shape)

    nn = BayesianNeuralNetworkVI(input_dim=d, output_dim=1, batch_size_vi=10, hidden_layer_sizes=(32, 32), prior_weight=0.001,
                                 learn_likelihood=True)
    nn.add_data_points(x_train, y_train)

    n_iter_fit = 200  # 2000
    for i in range(200):
        nn.fit(log_period=100, num_iter_fit=n_iter_fit)
        from matplotlib import pyplot as plt

        pred = nn.predict(x_plot)
        lcb, ucb = nn.confidence_intervals(x_plot)
        # ucb = out_dist.mean + out_dist.variance
        # lcb = out_dist.mean - out_dist.variance
        plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), alpha=0.3)
        plt.plot(x_plot, pred.mean)
        plt.scatter(x_train, y_train)

        plt.show()

# if __name__ == '__main__':
#     import haiku as hk
#     from jax import numpy as jnp
#     import jax
#     class MyMod(hk.Module):
#         def fwd(self):
#             return hk.get_parameter("hi", [3], init=hk.initializers.Constant(1.2))
#
#     def impure():
#         return MyMod().fwd()
#
#     init, apply = hk.transform(impure)
#     param = init(None)
#
#     flat, structure = jax.tree_flatten(param)
#     flat = flat[0]
#     print(flat)
#     n_samples = 2
#     flatsamps = [jax.random.normal(jax.random.PRNGKey(1), (flat.size, n_samples)) + jnp.expand_dims(flat, -1)]
#     sampled_params = jax.tree_util.tree_unflatten(structure, flatsamps)
#
#     print(structure)
#     print("")
#     print(sampled_params)
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
#
