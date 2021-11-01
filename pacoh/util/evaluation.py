import numpy as np
import scipy.stats


def _calib_error(pred_dist_vectorized, test_ys):
    cdf_vals = pred_dist_vectorized.cdf(test_ys)

    if test_t_tensor.shape[0] == 1:
        test_t_tensor = test_ys.flatten()
        cdf_vals = cdf_vals.flatten()

    num_points = test_t_tensor.shape[0]
    conf_levels = torch.linspace(0.05, 1.0, 20)
    emp_freq_per_conf_level = torch.sum(cdf_vals[:, None] <= conf_levels, dim=0).float() / num_points

    calib_rmse = torch.sqrt(torch.mean((emp_freq_per_conf_level - conf_levels)**2))
    return calib_rmse


def _calib_error_chi2(pred_dist_vectorized, test_t_tensor):
    import scipy.stats
    z2 = (((pred_dist_vectorized.mean - test_t_tensor) / pred_dist_vectorized.stddev) ** 2).detach().numpy()
    f = lambda p: np.mean(z2 < scipy.stats.chi2.ppf(p, 1))
    conf_levels = np.linspace(0.05, 1, 20)
    accs = np.array([f(p) for p in conf_levels])
    return np.sqrt(np.mean((accs - conf_levels)**2))