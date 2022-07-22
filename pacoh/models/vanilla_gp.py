import jax
import jax.numpy as jnp
import jax.random
import haiku as hk

from pacoh.models.regression_base import RegressionModel
from pacoh.models.pure.pure_functions import construct_vanilla_gp_forward_fns
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.util.initialization import initialize_model


class Vanilla_GP(RegressionModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        kernel_outputscale: float = 2.0,
        kernel_lengthscale: float = 1.0,
        likelihood_variance: float = 0.05,
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        random_state: jax.random.PRNGKey = None,
    ):
        """
        A Regression Model wrapping a simple exact-inference Gaussian Process

        :param input_dim: The dimensionality of input points
        :param output_dim: The dimensionality of output points. Only output_dim = 1 is currently supported
        :param kernel_outputscale: The outputscale of the RBF Kernel
        :param kernel_lengthscale: The lengthscale of the RBF Kernel
        :param likelihood_variance: The variance of the label noise
        :param normalize_data: Whether to normalize the data in some way
        :param normalizer: The normalizer used for normalization, if none, no normalization will happen
        :param random_state: A random seed
        """
        if output_dim > 1:
            raise NotImplementedError("GPs currently only support univariate mode")

        super().__init__(
            input_dim=input_dim,
            output_dim=1,
            normalize_data=normalize_data,
            normalizer=normalizer,
            random_state=random_state,
            flatten_ys=True,
        )
        factory = construct_vanilla_gp_forward_fns(
            input_dim,
            output_dim,
            kernel_outputscale,
            kernel_lengthscale,
            likelihood_variance,
            kernel_log_os_var=0.0,
            kernel_log_ls_var=0.0,
            likelihood_log_var=0.0,
        )

        # initialize model
        self._init_fn, self._apply_fns = hk.multi_transform_with_state(factory)
        self._rng, init_key = jax.random.split(self._rng)
        self._params, self._state = initialize_model(self._init_fn, init_key, (1, input_dim))

        # keep these around to reset to prior and evaluate prior
        self._prior_params = self._params
        self._prior_state = self._state

    @normalize_predict
    def predict(self, test_x):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)


        :param test_x: (ndarray) query input data of shape (n_samples, ndim_x)
        Notes:
            The decorator takes care of the mapping into and from the normalized space and adds return_density argument
        """
        self._rng, predict_key = jax.random.split(self._rng)
        pred_dist, self._state = self._apply_fns.base_learner_predict(
            self._params, self._state, predict_key, test_x
        )
        return pred_dist

    def reset_to_prior(self):
        self._params, self._state = self._prior_params, self._prior_state

    def fit(self):
        self._recompute_posterior()

    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints."""
        self._rng, fit_key = jax.random.split(self._rng)
        _, self._state = self._apply_fns.base_learner_fit(
            self._params, self._state, fit_key, self._xs_data, self._ys_data
        )

    @normalize_predict
    def _prior(self, xs: jnp.ndarray):
        """Returns the prior of the underlying GP for the given datapoints."""
        self._rng, predict_key = jax.random.split(self._rng)
        pred_dist, _ = self._apply_fns.base_learner_predict(
            self._prior_params, self._prior_state, predict_key, xs
        )
        return pred_dist


if __name__ == "__main__":
    # Some testing code
    import torch
    import numpy as np
    from matplotlib import pyplot as plt

    # generate some data
    n_train_samples = 20
    n_test_samples = 200
    torch.manual_seed(25)
    x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
    W = torch.tensor([[0.6]])
    b = torch.tensor([-1])
    y_data = (
        x_data.matmul(W.T)
        + torch.sin((0.6 * x_data) ** 2)
        + b
        + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))
    )
    y_data = torch.reshape(y_data, (-1,))

    x_data_train, x_data_test = (
        x_data[:n_train_samples].numpy(),
        x_data[n_train_samples:].numpy(),
    )
    y_data_train, y_data_test = (
        y_data[:n_train_samples].numpy(),
        y_data[n_train_samples:].numpy(),
    )

    # normalizer = DataNormalizer.from_dataset(x_data_test, y_data_test, normalize_data=True)

    gp = Vanilla_GP(input_dim=x_data.shape[-1], normalizer=None, normalize_data=True)
    gp.add_data_points(x_data_train, y_data_train)

    x_plot = np.linspace(6, -6, num=n_test_samples)

    pred_mean, _ = gp.predict(x_plot, return_density=False)
    lcb, ucb = gp.confidence_intervals(x_plot, confidence=0.9)
    plt.plot(x_plot, pred_mean)
    plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.4)

    gp.reset_to_prior()
    pred_mean, _ = gp.predict(x_plot, return_density=False)
    plt.plot(x_plot, pred_mean)

    plt.scatter(x_data_test, y_data_test)
    plt.scatter(x_data_train, y_data_train)
    plt.show()
