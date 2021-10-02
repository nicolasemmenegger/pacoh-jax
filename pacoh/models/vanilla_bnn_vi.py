import functools
import warnings
from typing import NamedTuple, Any, TypeVar, Callable, Dict

import jax
import numpy as np
import numpyro.distributions
import optax
import haiku as hk

from jax import numpy as jnp
from tqdm import trange

from pacoh.modules.abstract import RegressionModel
from pacoh.modules.data_handling import Sampler
from pacoh.modules.priors_posteriors import GaussianBelief, GaussianBeliefState
from pacoh.modules.util import pytree_unstack, _handle_batch_input_dimensionality, Tree, broadcast_params, pytree_shape, \
    stack_distributions
from pacoh.modules.bnn.batched_modules import transform_and_batch_module, multi_transform_and_batch_module
from pacoh.modules.distributions import JAXGaussianLikelihood


@transform_and_batch_module
def get_pure_batched_nn_functions(output_dim, hidden_layer_sizes, activation):
    def nn_forward(xs):
        nn = hk.nets.MLP(output_sizes=hidden_layer_sizes + (output_dim,), activation=activation)
        return nn(xs)

    return nn_forward


class LikelihoodInterface(NamedTuple):
    log_prob: Any
    get_posterior_from_means: Any


@functools.partial(multi_transform_and_batch_module, num_data_args={'log_prob': 2, 'get_posterior_from_means': 1})
def get_pure_batched_likelihood_functions(likelihood_initial_std):
    def factory() -> LikelihoodInterface:
        likelihood = JAXGaussianLikelihood(variance=likelihood_initial_std*likelihood_initial_std)  # TODO: non-constant variance over dimensions

        def log_prob(ys_true, ys_pred):
            return likelihood.log_prob(ys_true, ys_pred)

        def get_posterior_from_means(ys_pred):  # add noise to a mean prediction (same as add_noise with a zero_variance_pred_f)
            return likelihood.get_posterior_from_means(ys_pred)

        return get_posterior_from_means, LikelihoodInterface(log_prob=log_prob,
                                                             get_posterior_from_means=get_posterior_from_means)

    return factory


# def sample_gaussian_tree(mean_tree, std_tree, key, n_samples):
#     num_params = len(jax.tree_util.tree_leaves(mean_tree))  # number of prng keys we need
#     key, *keys = jax.random.split(key, num_params + 1)  # generate some key leaves
#     keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(mean_tree),
#                                              keys)  # build tree structure with keys
#
#     def sample_leaf(mean, std, key):
#         mean = jnp.expand_dims(mean, -1)
#         std = jnp.expand_dims(std, -1)
#
#         return (mean + jax.random.normal(key, (n_samples, *mean.shape), dtype=jnp.float32) * std).squeeze(axis=-1)
#
#     return jax.tree_multimap(sample_leaf, mean_tree, std_tree, keys_tree)


def neg_elbo(posterior: Dict[str, GaussianBeliefState],
             key: jax.random.PRNGKey,
             x_batch: jnp.array,
             y_batch: jnp.array,
             prior: Dict[str, GaussianBeliefState],
             prior_weight: float,
             batch_size_vi: int,
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
    log_likelihoods = likelihood_applys.log_prob(lh_params_batch, None, ys_true_rep, ys_pred)
    avg_log_likelihood = jnp.mean(log_likelihoods)

    # kl computation (regularization)
    avg_kl_divergence = GaussianBelief.log_prob(posterior['nn'], nn_params_batch) \
                        - GaussianBelief.log_prob(prior['nn'], nn_params_batch)

    if learn_likelihood:
        avg_kl_divergence = GaussianBelief.log_prob(posterior['lh'], lh_params_batch) \
                            - GaussianBelief.log_prob(prior['lh'], lh_params_batch)

    # result computation
    return - avg_log_likelihood + avg_kl_divergence * prior_weight



class BayesianNeuralNetworkVI(RegressionModel):
    def __init__(self, input_dim: int, output_dim: int, normalize_data: bool = True,
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
        :param likelihood_prior_mean: TODO figure out why this is not the same thing as the likelihood std
        :param likelihood_prior_std: The sigma in sigmall sim N(0,sigma)
        :param batch_size_vi: The number of samples from the posterior to approximate the expectation in the ELBO
        :param batch_size: The number of data points you get while training and predicting
        :param lr: The learning rate to use with the ELBO gradients
        """
        super().__init__(input_dim, normalize_data, random_state)
        self.output_dim = output_dim
        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size
        self.batch_size_vi = batch_size_vi

        """ A) Get batched forward functions for the nn and likelihood"""
        keys = jax.random.split(self._rds, self.batch_size_vi + 1)
        self._rds = keys[0]
        init_keys = keys[1:]
        batched_nn_init_fn, self.batched_nn_apply_fn, _ = get_pure_batched_nn_functions(output_dim,
                                                                                        hidden_layer_sizes,
                                                                                        activation)

        # if non learnable, this should be a REAL initialization TODO
        batched_likelihood_init_fn, self.batched_likelihood_apply_fns, self.batched_likelihood_apply_fns_batch_inputs = get_pure_batched_likelihood_functions(likelihood_prior_mean)

        """ B) Get dummy pytrees for the different module parameters => TODO somehow abstract this inside some utils. """
        nn_input = jnp.zeros((self.batch_size_vi, self.batch_size, input_dim))
        nn_params = batched_nn_init_fn(init_keys, nn_input)
        nn_param_template = pytree_unstack(nn_params)[0]  # just care about the form of the first models params

        likelihood_input = jnp.zeros((self.batch_size_vi, output_dim))
        likelihood_params = batched_likelihood_init_fn(init_keys, likelihood_input)
        likelihood_param_template = pytree_unstack(likelihood_params)[0]

        """ C) Setup prior and posterior modules """
        nn_prior_state = GaussianBeliefState.initialize(0.0, prior_std, nn_param_template)

        self.prior = {
            'nn': nn_prior_state,
        }
        self.posterior = {
            'nn': nn_prior_state.copy(),
        }

        self.learn_likelihood = learn_likelihood
        if learn_likelihood:
            lh_prior_state = GaussianBeliefState.initialize(likelihood_prior_mean, likelihood_prior_std, likelihood_param_template)
            self.prior['lh'] = lh_prior_state
            self.posterior['lh'] = lh_prior_state.copy()
            self.fixed_likelihood_params = None
        else:
            self.fixed_likelihood_params = likelihood_params
            # TODO make sure this gets initialized as desired
        # TODO understand why Jonas initializes the prior differently for learnable and non-learnable likelihood?

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

        self.elbo_val_and_grad = jax.jit(jax.value_and_grad(elbo_fn))

        self.batched_nn = jax.jit(self.batched_nn_apply_fn)

    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints. """
        # TODO ask Jonas what about recomputing the posterior -> start from scratch?  Not in an online setting I suppose
        warnings.warn("In this scenario, _recompute_postior does nothing for now, need to call fit explicitly")
        pass

    """ TODO: this could be put in the base class maybe?"""

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        train_batch_sampler = self._get_batch_sampler(self.xs_data, self.ys_data, self.batch_size)
        loss_list = []
        pbar = trange(num_iter_fit)
        for i in pbar:
            x_batch, y_batch = next(train_batch_sampler)
            loss = self._step(x_batch, jnp.expand_dims(y_batch, axis=-1))
            loss_list.append(loss)

            # if i % log_period == 0:
            #     loss = tf.reduce_mean(tf.convert_to_tensor(loss_list)).numpy()
            #     loss_list = []
            #     message = dict(loss=loss)
            #     if x_val is not None and y_val is not None:
            #         metric_dict = self.eval(x_val, y_val)
            #         message.update(metric_dict)
            #     pbar.set_postfix(message)

    def _get_batch_sampler(self, xs, ys, batch_size):
        # iterator that shuffles and repeats the data
        xs, ys = _handle_batch_input_dimensionality(xs, ys)
        num_train_points = xs.shape[0]

        if batch_size == -1:
            batch_size = num_train_points
        elif batch_size > 0:
            pass
        else:
            raise AssertionError('batch size must be either positive or -1')

        self._rds, sampler_key = jax.random.split(self._rds)
        return Sampler(xs, ys, batch_size, sampler_key)

    def predict(self, xs, num_posterior_samples=20):
        # a) data handling
        xs = _handle_batch_input_dimensionality(xs)
        warnings.warn("normalize")
        # xs = self._normalize_data(xs)

        # b) nn prediction
        self._rds, nn_key, lh_key = jax.random.split(self._rds, 3)

        nn_params = GaussianBelief.rsample(self.posterior['nn'], nn_key, num_posterior_samples)
        nn_shape = pytree_shape(nn_params)


        ys_pred = self.batched_nn(nn_params, None, xs)
        mean_shape = ys_pred.shape

        if self.learn_likelihood:
            lh_params = GaussianBelief.rsample(self.posterior['lh'], lh_key, num_posterior_samples)
        else:
            # lh_params = self.fixed_likelihood_params # there are not enough here
            lh_params = broadcast_params(pytree_unstack(self.fixed_likelihood_params, 1)[0], num_posterior_samples)

        lh_shape = pytree_shape(lh_params)

        pred_dists = self.batched_likelihood_apply_fns_batch_inputs.get_posterior_from_means(lh_params, None, ys_pred)
        # shape is (num_posterior_samples, batch_size, output_dim) -> transformed to (batch_size, output_dim, num_posterior_samples)
        pred_dists = stack_distributions(pred_dists)

        mixture = numpyro.distributions.Categorical(probs=jnp.ones((num_posterior_samples,)) / num_posterior_samples)


        pred_mixture = numpyro.distributions.MixtureSameFamily(mixture, pred_dists)

        # TODO handle
        #ys_pred = self._unnormalize_preds(ys_pred)  # should that be some kind of mean??
        #pred_dist = self._unnormalize_predictive_dist(pred_mixture)

        return jnp.mean(ys_pred, axis=0), pred_mixture


    def _step(self, x_batch, y_batch):
        self._rds, step_key = jax.random.split(self._rds)
        nelbo, gradelbo = self.elbo_val_and_grad(self.posterior, step_key, x_batch, y_batch)
        updates, new_opt_state = self.optimizer.update(gradelbo, self.optimizer_state, self.posterior)
        self.posterior = optax.apply_updates(self.posterior, updates)
        return nelbo


if __name__ == '__main__':
    np.random.seed(0)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200

    x_val = np.linspace(-4, 4, num=n_val)
    x_val = np.expand_dims(x_val, -1)
    y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)

    nn = BayesianNeuralNetworkVI(input_dim=d, output_dim=1, hidden_layer_sizes=(32, 32), prior_weight=0.001,
                                 learn_likelihood=False)
    nn.add_data_points(x_train, y_train)

    n_iter_fit = 2000  # 2000
    for i in range(10):
        nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
        from matplotlib import pyplot as plt


        pred, pred_mixture = nn.predict(x_val)

        plt.plot(x_val, pred_mixture.mean)
        ucb = pred + pred_mixture.variance
        lcb = pred - pred_mixture.variance
        plt.fill_between(x_val.flatten(), lcb.flatten(), ucb.flatten())
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
# i
