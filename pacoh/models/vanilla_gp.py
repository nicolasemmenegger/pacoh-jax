import jax
import jax.numpy as jnp
import jax.random
import haiku as hk

from pacoh.modules.distributions import AffineTransformedDistribution
from pacoh.models.regression_base import RegressionModel
from pacoh.modules.pure_functions import construct_vanilla_gp_forward_fns
from pacoh.util.data_handling import handle_batch_input_dimensionality, DataNormalizer
from pacoh.util.initialization import initialize_model

class GPRegressionVanilla(RegressionModel):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_outputscale: float = 2.0,
                 kernel_lengthscale: float = 1.0,
                 likelihood_variance: float = 0.05,
                 normalize_data: bool = True,
                 normalizer: DataNormalizer = None,
                 random_state: jax.random.PRNGKey = None):
        """
        A Regression Model wrapping a simple exact-inference Gaussian Process

        Args:
            input_dim: the input dimensionality of each data point.

        Keyword Args:
            kernel_outputscale: The outputscale of the RBF Kernel (i.e. linear scaling of the kernel's output.
            kernel_lengthscale: The lengthscale of the RBF Kernel, i.e. corrsponding to the standard
                deviation in the Gaussian density function.
            likelihood_variance: The variance of the Gaussian likelihood, i.e. the variance of the noise.
            normalize_data:  Whether to normalize the training data before fitting and the testing data before
                inference.
            normalization_stats: A dictionary containing the normalization statistics to use before fitting and
                inference. The dictionary should have the the keys x_mean, x_std, y_mean, y_std, of type np.ndarray, but
                with the
            random_state: A pseudo random number generator key from which to derive further
        """

        # Set up the boilerplate RegressionModel
        super().__init__(input_dim=input_dim, output_dim=output_dim, normalize_data=normalize_data,
                         normalizer=normalizer, random_state=random_state)
        factory = construct_vanilla_gp_forward_fns(input_dim,
                                                   kernel_outputscale,
                                                   kernel_lengthscale,
                                                   likelihood_variance)

        # initialize model
        self._init_fn, self._apply_fns = hk.multi_transform_with_state(factory)
        self._rng, init_key = jax.random.split(self._rng)
        self._params, self._state = initialize_model(self._init_fn, init_key, (1, input_dim))

        # keep these around to reset to prior and evaluate prior
        self._prior_params = self._params
        self._prior_state = self._state


    # @normalize_data_predict
    def predict(self, test_x, return_density=False, **kwargs):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)

        Notes:
            This includes both the epistemic and aleatoric uncertainty
        """
        test_x_normalized = self._normalizer.handle_data(test_x)
        self._rng, predict_key = jax.random.split(self._rng)
        pred_dist, self._state = self._apply_fns.pred_dist_fn(self._params, self._state, predict_key, test_x_normalized)

        pred_dist_transformed = AffineTransformedDistribution(pred_dist,
                                                              normalization_mean=self._normalizer.y_mean,
                                                              normalization_std=self._normalizer.y_std)
        if return_density:
            return pred_dist_transformed.iid_normal
        else:
            return pred_dist_transformed.mean, pred_dist_transformed.stddev

    def reset_to_prior(self):
        self._params, self._state = self._prior_params, self._prior_state

    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints. """
        self._rng, fit_key = jax.random.split(self._rng)
        _, self._state = self._apply_fns.fit_fn(self._params, self._state, fit_key, self.xs_data, self.ys_data)

    def _prior(self, xs: jnp.ndarray, return_density=True):
        """Returns the prior of the underlying GP for the given datapoints. """
        self._normalize_data(xs)
        self._rng, predict_key = jax.random.split(self._rng)
        pred_dist, _ = self._apply_fns.pred_dist_fn(self._prior_params, self._prior_state, predict_key, xs)

        pred_dist_transformed = AffineTransformedDistribution(pred_dist,
                                                              normalization_mean=self.y_mean,
                                                              normalization_std=self.y_std)

        return pred_dist_transformed



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
    y_data = x_data.matmul(W.T) + torch.sin((0.6 * x_data) ** 2) + b + torch.normal(mean=0.0, std=0.1, size=(
    n_train_samples + n_test_samples, 1))
    y_data = torch.reshape(y_data, (-1,))

    x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
    y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

    gp = GPRegressionVanilla(input_dim=x_data.shape[-1], output_dim=1)
    gp.add_data(x_data_train, y_data_train)

    x_plot = np.linspace(6, -6, num=n_test_samples)

    pred_dist = gp.predict(x_plot, return_density=True)
    lcb, ucb = gp.confidence_intervals(x_plot, confidence=0.9)
    # pred_dist = gp._affine_transformed_distribution_to_normal(pred_dist)
    pred_mean, pred_std = pred_dist.mean, pred_dist.scale
    # lcb, ucb = pred_mean-pred_std, pred_mean + pred_std
    # lcb, ucb = lcb.flatten(), ucb.flatten()
    plt.plot(x_plot, pred_mean)
    plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.4)

    gp.reset_to_prior()
    pred_mean, _ = gp.predict(x_plot, return_density=False)
    plt.plot(x_plot, pred_mean)

    plt.scatter(x_data_test, y_data_test)
    plt.scatter(x_data_train, y_data_train)
    plt.show()
