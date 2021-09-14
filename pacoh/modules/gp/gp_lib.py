import functools

import jax.lax.linalg
import numpyro.distributions
from abc import ABC, abstractmethod
from jax import numpy as jnp, vmap
from jax.scipy.linalg import cho_solve, cho_factor
import haiku as hk

class JAXExactGP:
    """
        A simple implementation of a gaussian process module with exact inference and Gaussian Likelihood
    """
    def __init__(self, mean_module, cov_module, likelihood):
        """
        Args:
            mean_module: hk.Module
            cov_module: hk.Module
            likelihood: hk.Module

        :param mean_module:
        :param cov_module:
        :param likelihood:
        """
        self.mean_module = mean_module
        self.likelihood = likelihood

        self.cov_vec_vec = cov_module
        self.cov_vec_set = hk.vmap(cov_module.__call__, (None, 0))
        self.cov_set_set = hk.vmap(self.cov_vec_set.__call__, (0, None))
        self.mean_set = hk.vmap(self.mean_module.__call__)

    def init_fn(self, dummy_xs):
        """Haiku initialiser"""
        hk.set_state("xs", jnp.zeros((0, 1), dtype=jnp.float32))
        hk.set_state("ys", jnp.zeros((0,), dtype=jnp.float32))
        hk.set_state("cholesky", (jnp.zeros((0, 0), dtype=jnp.float32), True))
        return self.likelihood(self._prior(dummy_xs))


    def _ys_centered(self, xs, ys):
        return ys - self.mean_set(xs).flatten()

    def _data_cov_with_noise(self, xs):
        data_cov = self.cov_set_set(xs, xs)
        return data_cov + jnp.eye(*data_cov.shape) * self.likelihood.variance()

    def fit(self, xs: jnp.ndarray, ys: jnp.ndarray) -> None:
        """ Fit adds the data in any case, stores cholesky if we are at eval stage for later reuse"""
        hk.set_state("xs", jnp.array(xs))
        hk.set_state("ys", jnp.array(ys))
        data_cov_w_noise = self._data_cov_with_noise(xs)
        hk.set_state("cholesky", cho_factor(data_cov_w_noise, lower=True)[0])

    @functools.partial(hk.vmap, in_axes=(None, 0))
    def posterior(self, x: jnp.ndarray):
        xs = hk.get_state("xs", dtype=jnp.float32)
        ys = hk.get_state("ys", dtype=jnp.float32)
        chol = hk.get_state("cholesky", dtype=jnp.float32)

        def has_data(operand):
            # we have data
            x, xs, chol = operand
            new_cov_row = self.cov_vec_set(x, xs)
            mean = self.mean_module(x) + jnp.dot(new_cov_row, cho_solve((chol, True), self._ys_centered(xs, ys)))
            var = self.cov_vec_vec(x, x) - jnp.dot(new_cov_row, cho_solve((chol, True), new_cov_row))
            std = jnp.sqrt(var)
            return numpyro.distributions.Normal(loc=mean, scale=std)

        def no_data(operand):
            x, _, __ = operand
            return self._prior(x)

        # TODO warning: this is quite confusing. I had this as a jax.lax.cond/hk.cond, but because the distinction
        # is on the size of xs, we cannot use these constructs, because they exactly assume that the size of
        # the inputs of both branches is the same
        if xs.size > 0:
            return has_data((x, xs, chol))
        else:
            return no_data((x, xs, chol))

    def pred_dist(self, xs_test):
        """prediction with noise"""
        predictive_dist_noiseless = self.posterior(xs_test)
        return self.likelihood(predictive_dist_noiseless)

    @functools.partial(hk.vmap, in_axes=(None, 0))
    def prior(self, x):
        return self._prior(x)

    def _prior(self, x):
        mean = self.mean_module(x)
        stddev = jnp.sqrt(self.cov_vec_vec(x, x)) # I am slightly surprised by this result
        return numpyro.distributions.Normal(loc=mean, scale=stddev)

    def add_data_and_refit(self, xs, ys):
        old_xs = hk.get_state("xs")
        old_ys = hk.get_state("ys")
        all_xs = jnp.concatenate([old_xs, xs])
        all_ys = jnp.concatenate([old_ys, ys])
        self.fit(all_xs, all_ys)

    """ ---- training utilities ---- """
    def marginal_ll(self, xs, ys):
        # computes the marginal log-likelihood of ys given xs and a posterior
        # computed on (xs,ys). This is differentiable and uses no stats
        ys_centered = self._ys_centered(xs, ys)
        data_cov_w_noise = self._data_cov_with_noise(xs)  # more noise
        cholesky = cho_factor(data_cov_w_noise)
        alpha = cho_solve(cholesky, ys_centered)
        ll = -0.5 * jnp.dot(ys_centered, alpha)
        ll -= jnp.sum(jnp.diag(cholesky[0]))  # this should be faster than trace
        ll -= xs.shape[0] / 2.0 * jnp.log(2.0 * jnp.pi)
        return ll

class JAXMean(hk.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, x):
        pass


class JAXConstantMean(JAXMean):
    """Mean module with a learnable mean. """
    def __init__(self, initial_constant=0.0):
        super().__init__()
        self.init_constant = initial_constant

    def __call__(self, x):
        # works for both batch or unbatched
        mean = hk.get_parameter("mu", shape=[], dtype=jnp.float32,
                                init=hk.initializers.Constant(self.init_constant))
        return jnp.ones(x.shape, dtype=jnp.float64) * mean


class JAXZeroMean(JAXMean):
    """Always zero, not learnable. """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return jnp.zeros(x.shape)