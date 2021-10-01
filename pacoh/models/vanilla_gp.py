import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import numpyro.distributions

from pacoh.modules import JAXExactGP, JAXConstantMean
from pacoh.modules.gp.kernels import JAXRBFKernel
from pacoh.modules import AffineTransformedDistribution, JAXGaussianLikelihood
from pacoh.modules.abstract import RegressionModel
from pacoh.modules.util import _handle_batch_input_dimensionality
from typing import Dict, NamedTuple, Any
import haiku as hk

class VanillaInterface(NamedTuple):
    fit_fn: Any
    pred_dist_fn: Any
    prior_fn: Any

def construct_vanilla_gp_forward_fns(input_dim, kernel_outputscale, kernel_lengthscale, likelihood_variance):
    def factory():
        # Initialize the mean module with a zero-mean, and use an RBF kernel with *no* learned feature map
        mean_module = JAXConstantMean(0.0)
        covar_module = JAXRBFKernel(input_dim, kernel_lengthscale, kernel_outputscale)
        likelihood = JAXGaussianLikelihood(likelihood_variance)
        gp = JAXExactGP(mean_module,
                        covar_module,
                        likelihood)

        # choose gp.pred_dist as the template function, as it includes the likelihood
        return gp.init_fn, VanillaInterface(fit_fn=gp.fit, pred_dist_fn=gp.pred_dist, prior_fn=gp.prior)

    return factory


class GPRegressionVanilla(RegressionModel):
    def __init__(self,
                 input_dim: int,
                 kernel_outputscale: float = 2.0,
                 kernel_lengthscale: float = 1.0,
                 likelihood_variance: float = 0.05,
                 normalize_data: bool = True,
                 normalization_stats: Dict[str, np.ndarray] = None,
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
        super().__init__(input_dim=input_dim, normalize_data=normalize_data, random_state=random_state)
        self._set_normalization_stats(normalization_stats)
        factory = construct_vanilla_gp_forward_fns(input_dim,
                                                   kernel_outputscale,
                                                   kernel_lengthscale,
                                                   likelihood_variance)

        init_fn, self._apply_fns = hk.multi_transform_with_state(factory)
        self._rds, init_key = jax.random.split(self._rds)
        # initialize parameters using some dummy data
        self._params, self._state = init_fn(init_key, jnp.ones((1, input_dim)))

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
        test_x = _handle_batch_input_dimensionality(test_x)
        test_x_normalized = self._normalize_data(test_x)
        self._rds, predict_key = jax.random.split(self._rds)
        pred_dist, self._state = self._apply_fns.pred_dist_fn(self._params, self._state, predict_key, test_x_normalized)
        pred_dist_transformed = AffineTransformedDistribution(pred_dist,
                                                              normalization_mean=self.y_mean,
                                                              normalization_std=self.y_std)
        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.scale
            return pred_mean, pred_std

    def reset_to_prior(self):
        raise NotImplementedError
        self._reset_data()
        self.gp.reset_to_prior()

    def _recompute_posterior(self):
        """Fits the underlying GP to the currently stored datapoints. """
        self._rds, fit_key = jax.random.split(self._rds)
        _, self._state = self._apply_fns.fit_fn(self._params, self._state, fit_key, self.xs_data, self.ys_data)

    def _prior(self, xs: jnp.ndarray):
        """Returns the prior of the underlying GP for the given datapoints.

        Args:
            xs: The datapoints to evaluate the prior on, of size (batch_size, input_dim).

        Returns:
            The prior distributions
        """
        return self._prior_fn(xs)


    # def predict_mean_std(self, test_x):
    #     # do we really need this?
    #     return self.predict(test_x, return_density=False)

    def state_dict(self):
        # TODO rename
        state_dict = {
            'model': self.gp.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.gp.load_state_dict(state_dict['model'])

    def _vectorize_pred_dist(self, pred_dist):
        # this is because ultimately, we do not want a TransformedDistribution object
        return numpyro.distributions.Normal(pred_dist.mean.flatten(), pred_dist.stddev.flatten())


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
    y_data = x_data.matmul(W.T) + torch.sin((0.6 * x_data)**2) + b + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))
    y_data = torch.reshape(y_data, (-1,))

    x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
    y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

    gp_mll = GPRegressionVanilla(input_dim=x_data.shape[-1])
    gp_mll.add_data(x_data_train, y_data_train)

    x_plot = np.linspace(6, -6, num=n_test_samples)

    pred_dist = gp_mll.predict(x_plot, return_density=True)
    pred_mean, pred_std = pred_dist.mean, pred_dist.stddev
    lcb, ucb = pred_mean-pred_std, pred_mean + pred_std
    lcb, ucb = lcb.flatten(), ucb.flatten()
    plt.plot(x_plot, pred_mean)
    plt.fill_between(x_plot, lcb, ucb, alpha=0.4)

    plt.scatter(x_data_test, y_data_test)
    plt.scatter(x_data_train, y_data_train)
    plt.show()