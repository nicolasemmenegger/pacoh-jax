import numpy as np
import scipy.stats

from jax import numpy as jnp
from pacoh.util.distributions import is_diagonal_gaussian_dist


def calib_error(pred_dist_diagonalized, test_ys):
    """Both inputs should be 2 dimensional."""
    if not is_diagonal_gaussian_dist(pred_dist_diagonalized):
        raise NotImplementedError("Cannot compute for something that is not a diagonal gaussian")

    cdf_vals = pred_dist_diagonalized.base_dist.cdf(test_ys)
    if not (cdf_vals.shape[-1] == 1 and cdf_vals.ndim == 2):
        cdf_vals = np.expand_dims(cdf_vals, axis=1)
    num_points = test_ys.shape[0]
    conf_levels = np.linspace(0.05, 1.0, 20)
    conf_levels = np.expand_dims(conf_levels, axis=0)

    compare_matrix = cdf_vals <= conf_levels
    # compare_matrix[point, quantile] will be true if the model thinks the probability of f(point_x) <= point_y is smaller than quantily
    emp_freq_per_conf_level = np.sum(compare_matrix, axis=0) / num_points
    # emp_freq_per_confidence_level[quantile] The probability that the model predicts a probability lower than the qth quantile
    # under the empirical distribution

    calib_rmse = np.sqrt(np.mean((emp_freq_per_conf_level - conf_levels) ** 2))
    return calib_rmse


def calib_error_chi2(pred_dist_diagonalized, test_ys):
    z2 = ((pred_dist_diagonalized.mean - test_ys) / jnp.sqrt(pred_dist_diagonalized.variance)) ** 2
    f = lambda p: np.mean(z2 < scipy.stats.chi2.ppf(p, 1))
    conf_levels = np.linspace(0.05, 1, 20)
    accs = np.array([f(p) for p in conf_levels])
    return np.sqrt(np.mean((accs - conf_levels) ** 2))
