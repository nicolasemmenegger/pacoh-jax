import warnings

import jax
import numpy as np
import optax
from jax import numpy as jnp
from tqdm import trange

from pacoh.algorithms.svgd import SVGD
from pacoh.models.vanilla_bnn_vi import get_pure_batched_nn_functions
from pacoh.models.regression_base import RegressionModel
from pacoh.modules.distributions import AffineTransformedDistribution, get_mixture
from pacoh.modules.kernels import pytree_rbf_set
from pacoh.modules.belief import GaussianBeliefState, GaussianBelief
from pacoh.models.pure.pure_functions import get_pure_batched_likelihood_functions
from pacoh.util.data_handling import handle_batch_input_dimensionality
from pacoh.util.initialization import initialize_batched_model


class BayesianNeuralNetworkSVGD(RegressionModel):

    def __init__(self, input_dim: int, output_dim: int, normalize_data: bool = True,
                 random_state: jax.random.PRNGKey = None,
                 hidden_layer_sizes=(32, 32, 32, 32), activation=jax.nn.elu,
                 likelihood_std=0.1, learn_likelihood=True, prior_std=0.1, prior_weight=1e-4,
                 likelihood_prior_mean=jnp.log(0.1), likelihood_prior_std=1.0,
                 n_particles=10, batch_size=8, bandwidth=100., lr=1e-3, sqrt_mode=False, meta_learned_prior=None, normalization_stats=None):

        super().__init__(input_dim, output_dim, normalize_data, normalization_stats, random_state)
        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.sqrt_mode = sqrt_mode
        self.learn_likelihood = learn_likelihood

        if not learn_likelihood:
            raise NotImplementedError("the learn_likelihood flag = False is not supported yet")


        """ A) Get batched forward functions for the nn and likelihood """
        self._rng, nn_init_key, lh_init_key = jax.random.split(self._rng, 3)
        batched_nn_init, self.batched_nn_apply_fn, _ = get_pure_batched_nn_functions(output_dim,hidden_layer_sizes, activation)
        lh_init, self.lh_apply_shared, self.lh_apply_broadcast = get_pure_batched_likelihood_functions(likelihood_prior_mean)

        """ B) Get pytrees with parameters for stacked models """
        nn_params, nn_params_template = initialize_batched_model(batched_nn_init, self.n_particles, nn_init_key, (self.batch_size, input_dim))
        likelihood_params, likelihood_params_template = initialize_batched_model(lh_init, self.n_particles, lh_init_key, (output_dim,))

        """ C) setup prior module """
        if meta_learned_prior is None:
            self.prior = {
                'nn': GaussianBeliefState.initialize(0.0, prior_std, nn_params_template)
            }

            if learn_likelihood:
                lh_prior_state = GaussianBeliefState.initialize(likelihood_prior_mean, likelihood_prior_std,
                                                                likelihood_params_template)
                self.prior['lh'] = lh_prior_state
                self.fixed_likelihood_params = None
            else:
                self.fixed_likelihood_params = likelihood_params

            self.meta_learned_prior_mode = False
        else:
            self.prior = meta_learned_prior
            self.meta_learned_prior_mode = True

        """ D) Sample initial particles of the posterior mixture """
        self._rng, sample_key = jax.random.split(self._rng)
        self.particles = GaussianBelief.rsample_multiple(self.prior, sample_key, n_particles)

        warnings.warn("the option to not learn the likelihood is currently not supported")
        if not self.learn_likelihood:
            self.particles['lh'] = likelihood_params  # TODO make sure it's not getting differentiated in that case

        warnings.warn("meta-learned prior mode not supported yet")
        # if self.meta_learned_prior_mode:
        #     # initialize posterior particles from meta-learned prior
        #     params = tf.reshape(self.prior.sample(n_particles // self.prior.n_batched_priors), (n_particles, -1))
        #     self.particles = tf.Variable(params)
        # else:
        #     # initialize posterior particles from model initialization
        #     nn_params = self.nn.get_variables_stacked_per_model()
        #     likelihood_params = tf.ones((self.n_particles, self.likelihood_param_size)) * likelihood_prior_mean
        #     self.particles = tf.Variable(tf.concat([nn_params, likelihood_params], axis=-1))

        # setup kernel and optimizer

        """ D) setup all the forward functions needed by the SVGD class. """
        def target_log_prob_batched(particles, rngs, *data):
            # predict
            xs, ys = data
            ys_pred = self.batched_nn_apply_fn(particles['nn'], None, xs)
            ys_true_rep = jnp.repeat(jnp.expand_dims(ys, axis=0), n_particles, axis=0)

            log_likelihoods = self.lh_apply_broadcast.log_prob(particles['lh'], None, ys_true_rep, ys_pred)
            avg_log_likelihood = jnp.mean(log_likelihoods)

            prior_log_prob = GaussianBelief.log_prob(self.prior['nn'], particles['nn'])
            prior_log_prob += GaussianBelief.log_prob(self.prior['lh'], particles['lh'])

            return prior_log_prob + avg_log_likelihood

        def kernel_fwd(particles):
            return pytree_rbf_set(particles, particles, length_scale=bandwidth, output_scale=1.0)


        """ E) Setup optimizer on particles """
        lr_scheduler = optax.constant_schedule(lr)
        self.optimizer = optax.adam(lr_scheduler)
        self.optimizer_state = self.optimizer.init(self.particles)
        self.svgd = SVGD(target_log_prob_batched, kernel_fwd, self.optimizer, self.optimizer_state)

    def _recompute_posterior(self):
        pass

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        train_batch_sampler = self._get_batch_sampler(self.xs_data, self.ys_data, self.batch_size)
        loss_list = []
        pbar = trange(num_iter_fit)
        for i in pbar:
            x_batch, y_batch = next(train_batch_sampler)
            loss = self.step(x_batch, jnp.expand_dims(y_batch, axis=-1))
            loss_list.append(loss)

            if i % log_period == 0:
                loss = jnp.mean(jnp.array(loss_list))
                loss_list = []
                message = dict(loss=loss)
                # if x_val is not None and y_val is not None:
                #     metric_dict = self.eval(x_val, y_val)
                #     message.update(metric_dict)
                pbar.set_postfix(message)

    def predict(self, xs, return_density=False, **kwargs):
        # a) data handling
        xs = handle_batch_input_dimensionality(xs)
        xs = self._normalize_data(xs)

        # b) nn prediction
        ys_pred = self.batched_nn_apply_fn(self.particles['nn'], None, xs)
        pred_dists = self.lh_apply_broadcast.get_posterior_from_means(self.particles['lh'], None, ys_pred)
        pred_mixture = get_mixture(pred_dists, self.n_particles)


        # c) normalization
        pred_mixture_transformed = AffineTransformedDistribution(pred_mixture,
                                                                 normalization_mean=self.y_mean,
                                                                 normalization_std=self.y_std)
        # ys_pred = self._unnormalize_preds(ys_pred)  # should that be some kind of mean??
        # pred_dist = self._unnormalize_predictive_dist(pred_mixture)

        if return_density:
            return pred_mixture_transformed
        else:
            return pred_mixture_transformed.mean, pred_mixture_transformed.stddev

    def step(self, x_batch, y_batch):
        self.svgd.step(self.particles, x_batch, y_batch)



# if __name__ == '__main__':
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
#     nn = BayesianNeuralNetworkSVGD(x_train, y_train, hidden_layer_sizes=(64, 64), prior_weight=0.001, bandwidth=1000.0)
#
#     n_iter_fit = 500
#     for i in range(10):
#         nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
#         if d == 1:
#             x_plot = tf.range(-8, 8, 0.1)
#             nn.plot_predictions(x_plot, show=True)


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

    nn = BayesianNeuralNetworkSVGD(input_dim=d, output_dim=1, hidden_layer_sizes=(32, 32), prior_weight=0.001,
                                   bandwidth=1000.0, learn_likelihood=True)

    nn.add_data_points(x_train, y_train)

    n_iter_fit = 500  # 2000
    for i in range(10):
        nn.fit(log_period=10, num_iter_fit=n_iter_fit)
        from matplotlib import pyplot as plt

        pred, pred_mixture = nn.predict(x_plot)

        ucb = pred + pred_mixture.variance
        lcb = pred - pred_mixture.variance
        plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), alpha=0.3)
        plt.plot(x_plot, pred_mixture.mean)
        plt.scatter(x_train, y_train)

        plt.show()