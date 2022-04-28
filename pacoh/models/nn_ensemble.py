import jax.nn
import optax
from jax import numpy as jnp
import haiku as hk
import numpy as np

from pacoh.models.pure.pure_functions import construct_bnn_forward_fns
from pacoh.models.pure.pure_interfaces import NNBaseLearner
from pacoh.models.regression_base import RegressionModel
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.util.distributions import get_mixture
from pacoh.modules.kernels import pytree_sq_l2_dist
from pacoh.util.constants import MLP_MODULE_NAME
from pacoh.util.data_handling import normalize_predict
from pacoh.util.initialization import (
    initialize_batched_model,
    initialize_optimizer,
    initialize_model,
)


def construct_bnn_forward_fn(
    output_dim,
    hidden_layer_sizes,
    activation,
    likelihood_initial_std,
    learn_likelihood=True,
):
    # this is only here for convenience. Not the same as construct_bnn_forward_fns
    def factory():
        likelihood_module = JAXGaussianLikelihood(
            variance=likelihood_initial_std * likelihood_initial_std,
            learn_likelihood=learn_likelihood,
        )

        nn = hk.nets.MLP(
            name=MLP_MODULE_NAME,
            output_sizes=hidden_layer_sizes + (output_dim,),
            activation=activation,
        )

        def pred_dist(xs):
            means = nn(xs)
            return likelihood_module(means)  # this adds heteroscedastic variance to all datapoints

        def log_prob(ys_true, ys_pred):
            return likelihood_module.log_prob(ys_true, ys_pred)

        def pred(xs):
            res = nn(xs)
            return res

        return pred_dist, NNBaseLearner(log_prob=log_prob, pred_dist=pred_dist, pred=pred)

    return factory


class SimpleNNTutorial(RegressionModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__(input_dim, output_dim)

        factory = construct_bnn_forward_fn(
            output_dim,
            (32, 32),
            jax.nn.sigmoid,
            1.0 / jnp.sqrt(2),
            learn_likelihood=False,
        )

        init, self.apply = hk.multi_transform(factory)

        # model and parameter init
        self._batch_size = 8
        self._rng, init_key = jax.random.split(self._rng)
        self.params = initialize_model(init, init_key, (self._batch_size, input_dim))
        self._lambda = 0.1

        # loss function init
        def loss_function(params, xs, ys_labels):
            pred_ys = self.apply.pred(params, None, xs)
            return -self.apply.log_prob(params, None, ys_labels, pred_ys)

        self._loss_val_and_grad = jax.jit(jax.value_and_grad(loss_function))
        self._alternate_loss = loss_function

        # optimizer init
        self.optimizer, self.optimizer_state = initialize_optimizer(
            "Adam", 0.001, self.params, lr_decay=1.0  # no decay
        )

    @normalize_predict
    def predict(self, test_xs):
        return self.apply.pred_dist(self.params, None, test_xs)

    def _step(self, xs_batch, ys_batch):
        loss, grad = self._loss_val_and_grad(self.params, xs_batch, ys_batch)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _recompute_posterior(self):
        pass


class EnsembleNNTutorial(RegressionModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__(input_dim, output_dim)

        init, self.apply, self.apply_bcst = construct_bnn_forward_fns(
            output_dim,
            (32, 32),
            jax.nn.sigmoid,
            0.1,
            learn_likelihood=False,
        )

        # model and parameter init
        self._batch_size = 8
        self._ensemble_size = 10
        self._rng, init_key = jax.random.split(self._rng)
        self.params, _ = initialize_batched_model(
            init, self._ensemble_size, init_key, (self._batch_size, input_dim)
        )
        self._lambda = 0.1

        # loss function init
        def loss_function(params, xs, ys_labels):
            pred_ys = self.apply.pred(params, None, xs)
            ys_true_rep = jnp.repeat(jnp.expand_dims(ys_labels, axis=0), self._ensemble_size, axis=0)
            return -jnp.sum(
                self.apply_bcst.log_prob(params, None, ys_true_rep, pred_ys)
            ) + self._lambda * pytree_sq_l2_dist(params, params)

        self._loss_val_and_grad = jax.value_and_grad(loss_function)

        # optimizer init
        self.optimizer, self.optimizer_state = initialize_optimizer(
            "AdamW", 0.001, self.params, lr_decay=1.0, weight_decay=0.1
        )

    @normalize_predict
    def predict(self, test_xs):
        components = self.apply.pred_dist(self.params, None, test_xs)
        return get_mixture(components, self._ensemble_size)  # None since no randomness needed in forward pass

    def _step(self, xs_batch, ys_batch):
        loss, grad = self._loss_val_and_grad(self.params, xs_batch, ys_batch)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def _recompute_posterior(self):
        pass


if __name__ == "__main__":
    np.random.seed(1)
    from jax.config import config
    from matplotlib import pyplot as plt

    config.update("jax_debug_nans", False)
    config.update("jax_disable_jit", False)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200

    x_plot = np.linspace(-4, 4, num=n_val)
    x_plot = np.expand_dims(x_plot, -1)
    y_val = np.sin(x_plot) + np.random.normal(scale=0.1, size=x_plot.shape)

    x_train, y_train, x_plot, y_val = (
        jnp.array(x_train),
        jnp.array(y_train),
        jnp.array(x_plot),
        jnp.array(y_val),
    )
    nn = EnsembleNNTutorial(
        input_dim=d,
        output_dim=1,
    )
    nn.add_data_points(x_train, y_train)

    print(nn._xs_data)
    print(nn._ys_data)
    n_iter_fit = 200
    for i in range(200):
        nn.fit(log_period=100, num_iter_fit=n_iter_fit, xs_val=x_plot, ys_val=y_val)

        pred = nn.predict(x_plot)
        lcb, ucb = nn.confidence_intervals(x_plot)
        plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), alpha=0.3)
        plt.plot(x_plot, pred.mean)
        plt.scatter(x_train, y_train)

        plt.show()
