import functools
import warnings
from collections import Callable

import jax
import numpy as np
import optax
from jax import numpy as jnp

from pacoh.algorithms.svgd import SVGD
from pacoh.models.regression_base import RegressionModel
from pacoh.modules.distributions import get_mixture
from pacoh.modules.kernels import pytree_rbf_set
from pacoh.modules.belief import GaussianBeliefState, GaussianBelief
from pacoh.models.pure.pure_functions import construct_bnn_forward_fns
from pacoh.util.constants import LIKELIHOOD_MODULE_NAME, MLP_MODULE_NAME
from pacoh.util.data_handling import normalize_predict, DataNormalizer
from pacoh.util.initialization import initialize_batched_model


class BayesianNeuralNetworkSVGD(RegressionModel):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 normalize_data: bool = True,
                 normalizer: DataNormalizer = None,
                 random_state: jax.random.PRNGKey = None,
                 hidden_layer_sizes=(32, 32),
                 activation: Callable = jax.nn.elu,
                 learn_likelihood: bool = True,
                 prior_std: float = 1.0,
                 prior_weight: float = 0.1,
                 likelihood_prior_mean: float = jnp.log(0.1),
                 likelihood_prior_std: float = 1.0,
                 n_particles: int = 10,
                 batch_size: int = 8,
                 bandwidth: float = 100.,
                 lr: float = 1e-3):
        # TODO what is sqrt_mode, meta_learned_prior_mode
        super().__init__(input_dim, output_dim, normalize_data, normalizer, random_state)
        self.batch_size = batch_size
        self.n_particles = n_particles

        # a) Get batched forward functions for the nn and likelihood (note that learn_likelihood controls whether the
        #    likelihood module actually contributes parameters
        self._rng, init_key = jax.random.split(self._rng)
        init, self.apply, self.apply_broadcast = construct_bnn_forward_fns(output_dim, hidden_layer_sizes,
                                                                           activation, likelihood_prior_mean,
                                                                           learn_likelihood)

        # b) Initialize the prior and the particles of the posterior
        params, template = initialize_batched_model(init, n_particles, init_key, (batch_size, input_dim))

        def mean_std_map(mod_name: str, _: str, __: jnp.array):
            if LIKELIHOOD_MODULE_NAME in mod_name:
                return likelihood_prior_mean, likelihood_prior_std
            elif MLP_MODULE_NAME in mod_name:
                return 0.0, prior_std
            else:
                raise AssertionError("Unknown hk.Module: can only handle mlp and likelihood")

        self.prior = GaussianBeliefState.initialize_heterogenous(mean_std_map, template)
        self._rng, particle_sample_key = jax.random.split(self._rng)
        self.particles = GaussianBelief.rsample(self.prior, particle_sample_key, n_particles)

        # c) setup all the forward functions needed by the SVGD class.
        def target_post_prob_batched(particles, rngs, *data, apply, apply_bdcst):
            xs, ys = data
            ys_pred = apply.pred(particles, None, xs)
            ys_true_rep = jnp.repeat(jnp.expand_dims(ys, axis=0), n_particles, axis=0)

            log_likelihoods = apply_bdcst.log_prob(particles, None, ys_true_rep, ys_pred)
            avg_log_likelihood = jnp.mean(log_likelihoods)

            prior_log_prob = GaussianBelief.log_prob(self.prior, particles)

            return prior_weight * prior_log_prob + avg_log_likelihood

        def kernel_fwd(particles):
            # TODO here we could actually transform the parameters in order to work with the canonical parametrization
            # and not with the logscale transforms used to ensure positivity
            return pytree_rbf_set(particles, particles, bandwidth, 1.0)


        # d) learning rate scheduler
        lr_scheduler = optax.constant_schedule(lr)
        self.optimizer = optax.adam(lr_scheduler)
        self.optimizer_state = self.optimizer.init(self.particles)
        log_prob = functools.partial(target_post_prob_batched, apply=self.apply, apply_bdcst=self.apply_broadcast)
        self.svgd = SVGD(log_prob, kernel_fwd, self.optimizer, self.optimizer_state)

    def _recompute_posterior(self):
        pass

    def fit(self, xs_val=None, ys_val=None, log_period=500, num_iter_fit=None):
        super().fit(xs_val, ys_val, log_period, num_iter_fit)

    @normalize_predict
    def predict(self, xs):
        return get_mixture(self.apply.pred_dist(self.particles, None, xs), self.n_particles)

    def _step(self, x_batch, y_batch):
        self.particles = self.svgd.step(self.particles, x_batch, y_batch)
        return 0.0 # here I could return the log likelihood somehow


if __name__ == '__main__':
    np.random.seed(1)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200

    x_plot = np.linspace(-6, 6, num=n_val)
    x_plot = np.expand_dims(x_plot, -1)
    y_val = np.sin(x_plot) + np.random.normal(scale=0.1, size=x_plot.shape)

    nn = BayesianNeuralNetworkSVGD(input_dim=d, output_dim=1, hidden_layer_sizes=(32, 32), prior_weight=0.001,
                                   bandwidth=1000.0, learn_likelihood=True)

    nn.add_data_points(x_train, y_train)

    n_iter_fit = 100  # 2000
    for i in range(200):
        nn.fit(log_period=10, num_iter_fit=n_iter_fit)
        from matplotlib import pyplot as plt

        pred = nn.predict(x_plot)
        lcb, ucb = nn.confidence_intervals(x_plot)
        plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), alpha=0.3)
        plt.plot(x_plot, pred.mean)
        plt.scatter(x_train, y_train)

        plt.show()