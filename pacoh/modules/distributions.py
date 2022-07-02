import numpyro
from jax import numpy as jnp
import haiku as hk

from pacoh.modules.common import PositiveParameter
from pacoh.util.constants import LIKELIHOOD_MODULE_NAME
from pacoh.util.distributions import get_diagonal_gaussian, is_diagonal_gaussian_dist


class JAXGaussianLikelihood(hk.Module):
    def __init__(
        self,
        variance: float = 1.0,
        log_var_std=0.0,
        variance_constraint_gt=0.0,
        output_dim=1,
        learn_likelihood=True,
    ):
        super().__init__(LIKELIHOOD_MODULE_NAME)
        variance = jnp.ones((output_dim,)) * variance
        if learn_likelihood:
            self.variance = PositiveParameter(
                mean=variance, log_stddev=log_var_std, boundary_value=variance_constraint_gt
            )
        else:
            self.variance = lambda: variance

    def __call__(self, posterior):
        if isinstance(posterior, jnp.ndarray):
            # just an array of means
            new_scale = jnp.sqrt(self.variance()) * jnp.ones_like(posterior)
            return get_diagonal_gaussian(posterior, new_scale)
        elif isinstance(posterior, numpyro.distributions.MultivariateNormal):
            d = posterior.loc.shape[0]
            cov_with_noise = posterior.covariance_matrix + jnp.eye(d, d) * self.variance()
            return numpyro.distributions.MultivariateNormal(
                loc=posterior.loc, covariance_matrix=cov_with_noise
            )
        elif is_diagonal_gaussian_dist(posterior):
            scale = jnp.sqrt(posterior.variance + self.variance())
            diag = get_diagonal_gaussian(posterior.mean, scale)
            return diag
        else:
            raise ValueError(
                "posterior should be either a multivariate diagonal, full covariance gaussian or an "
                + "array of means"
            )

    def log_prob(self, ys_true, ys_pred):
        gauss = get_diagonal_gaussian(
            jnp.zeros_like(ys_true),
            jnp.sqrt(self.variance()) * jnp.ones_like(ys_true),
        )
        return gauss.log_prob(ys_pred - ys_true)
