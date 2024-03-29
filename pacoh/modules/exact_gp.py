import functools

import numpyro.distributions
from numpyro.distributions import MultivariateNormal
from pacoh.util.distributions import get_diagonal_gaussian, diagonalize_gaussian

from jax import numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
import haiku as hk


class JAXExactGP:
    """
    A simple implementation of a gaussian process module with exact inference and Gaussian Likelihood
    """

    def __init__(self, mean_module, cov_module, likelihood):
        """
        :param mean_module: Should have a call function with signature (*, input_dim) -> (*, output_dim) where *
            may be empty
        :param cov_module: Should implement a function (input_dim,) x (input_dim,) -> float
        :param likelihood:
        """
        self.mean_module = mean_module
        self.likelihood = likelihood

        self.cov_vec_vec = cov_module
        self.cov_vec_set = hk.vmap(cov_module.__call__, (None, 0), split_rng=False)
        self.cov_set_set = hk.vmap(self.cov_vec_set.__call__, (0, None), split_rng=False)

    def init_fn(self, dummy_xs):
        """Haiku initialiser"""
        hk.set_state("xs", jnp.zeros((0, 1), dtype=jnp.float32))
        hk.set_state("ys", jnp.zeros((0,), dtype=jnp.float32))
        hk.set_state("cholesky", jnp.zeros((0, 0), dtype=jnp.float32))
        return self.likelihood(self.prior(dummy_xs))

    def _ys_centered(self, xs, ys):
        return ys.flatten() - self.mean_module(xs).flatten()

    def _data_cov_with_noise(self, xs):
        data_cov = self.cov_set_set(xs, xs)
        return data_cov + jnp.eye(*data_cov.shape) * self.likelihood.variance()

    def fit(self, xs: jnp.ndarray, ys: jnp.ndarray) -> None:
        """Fit adds the data in any case, stores cholesky if we are at eval stage for later reuse"""
        hk.set_state("xs", jnp.array(xs))
        hk.set_state("ys", jnp.array(ys))
        data_cov_w_noise = self._data_cov_with_noise(xs)
        hk.set_state("cholesky", cho_factor(data_cov_w_noise, lower=True)[0])

    def posterior(self, xs_test: jnp.ndarray, return_full_covariance=True):
        xs = hk.get_state("xs", dtype=jnp.float32)
        ys = hk.get_state("ys", dtype=jnp.float32)
        chol = hk.get_state("cholesky", dtype=jnp.float32)

        if xs.size > 0:
            test_train_cov = self.cov_set_set(xs_test, xs)

            # mean prediction
            ys_cent = self._ys_centered(xs, ys).flatten()
            mean = self.mean_module(xs_test).flatten() + test_train_cov @ cho_solve((chol, True), ys_cent)

            cov = self.cov_set_set(xs_test, xs_test)
            cov -= test_train_cov @ cho_solve((chol, True), test_train_cov.T)
            assert cov.ndim == 2

            if return_full_covariance:
                # full covariance information
                return numpyro.distributions.MultivariateNormal(mean, cov)
            else:
                return get_diagonal_gaussian(mean, jnp.diag(cov), 1)

                # return numpyro.Independent()

            # new_cov_row = self.cov_vec_set(x, xs)
            # ys_cent = self._ys_centered(xs, ys).flatten()
            # cho_sol = cho_solve((chol, True), ys_cent)
            # mean = self.mean_module(x) + jnp.dot(new_cov_row.flatten(), cho_sol)
            # var = self.cov_vec_vec(x, x) - jnp.dot(new_cov_row, cho_solve((chol, True), new_cov_row))
            # std = jnp.sqrt(var)
            # return numpyro.distributions.Normal(loc=mean, scale=jnp.ones((1,))*std)

        else:
            return self.prior(xs_test)

    def pred_dist(self, xs_test, noiseless=False):
        """Prediction with noise."""
        predictive_dist_noiseless = self.posterior(xs_test, return_full_covariance=True)
        if noiseless:
            return predictive_dist_noiseless
        else:
            return self.likelihood(predictive_dist_noiseless)

    def prior(self, xs, return_full_covariance=True):
        mean = self.mean_module(xs)
        cov = self.cov_set_set(xs, xs)
        if return_full_covariance:
            return MultivariateNormal(mean.flatten(), cov)
        else:
            return get_diagonal_gaussian(mean, jnp.diag(cov))

    def add_data_and_refit(self, xs, ys):
        old_xs = hk.get_state("xs")
        old_ys = hk.get_state("ys")
        all_xs = jnp.concatenate([old_xs, xs])
        all_ys = jnp.concatenate([old_ys, ys])
        self.fit(all_xs, all_ys)

    """ ---- training utilities ---- """

    def marginal_ll(self, xs, ys):
        data_cov_w_noise = self._data_cov_with_noise(xs)  # Kernel matrix with noise on diagonal
        ys_centered = self._ys_centered(xs, ys)  # ys - prior_mean_function(xs)
        mvn = numpyro.distributions.MultivariateNormal(jnp.zeros_like(xs.flatten()), data_cov_w_noise)
        alt_ll = mvn.log_prob(ys_centered)
        return alt_ll
