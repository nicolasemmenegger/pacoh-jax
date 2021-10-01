import warnings
from typing import Union

import numpy as np
from pacoh.modules.util import _handle_batch_input_dimensionality, _handle_point_input_dimensionality
from config import device

import jax
from jax import numpy as jnp
from abc import ABC, abstractmethod


class RegressionModel(ABC):
    def __init__(self, input_dim: int, normalize_data: bool = True, random_state: jax.random.PRNGKey = None):
        """
        Abstracts the boilerplate functionality of a Regression Model. This includes data normalization,
        fitting and inference
        Public Methods:
            add_data_point(s): add data and refit the posterior
            predict: returns the predictive distribution for new points
        Args:
            input_dim: The dimensions of the input features
            normalize_data: Whether to do data normalisation prior to fitting and inference
            random_state: The master PRNGKey to be used in downstream modules
        Notes:
            Subclasses need to implement the following:
                _fit_posterior(): A method that is called after adding data points
                predict(xs): target prediction
        """
        self.input_dim = input_dim
        self.normalize_data = normalize_data
        self.xs_data = jnp.zeros((0, self.input_dim), dtype=np.double)
        self.ys_data = jnp.zeros((0,), dtype=np.double)
        self._num_train_points = 0
        self._rds = random_state if random_state is not None else jax.random.PRNGKey(42)


    """ ------ Public Methods ------ """
    def add_data(self, xs: np.ndarray, ys: Union[float, np.ndarray]):
        """
        A method to either add a single data point or a batch of datapoints
        to the model
        Args:
            xs: the feature(s)
            ys: the label(s)
        Notes:
            The more explicit add_data_point and add_data_points are preferred
        """
        warnings.warn("TBD: add_data should be renamed as add_data_points and add_data_point, \
                       depending on usage in online or i.i.d. setting", PendingDeprecationWarning)
        if isinstance(ys, float) or ys.size == 1:
            self.add_data_point(xs, ys)
        else:
            self.add_data_points(xs, ys)

    def add_data_points(self, xs: np.ndarray, ys: np.ndarray):
        """
        Method to add some number of data points to the fitted regression model and refit.
        Args:
            xs: the points to add, of shape (num_data, self.input_dim) or (num_data) if input_dim == 1
            ys: the corresponding observations, of shape (num_data) or (num_data, 1)
        """
        assert xs.ndim == 2 and xs.shape[1] == self.input_dim
        # handle input dimensionality and normalize data
        xs, ys = _handle_batch_input_dimensionality(xs, ys)
        #xs, ys = self._normalize_data(xs, ys)
        warnings.warn("undo")
        # concatenate to new datapoints
        self.xs_data = np.concatenate([self.xs_data, xs])
        self.ys_data = np.concatenate([self.ys_data, ys])
        self._num_train_points += ys.shape[0]

        assert self.xs_data.shape[0] == self.ys_data.shape[0]
        assert self._num_train_points == 1 or self.xs_data.shape[0] == self._num_train_points

        self._recompute_posterior()

    def add_data_point(self, x, y):
        """
        Method to add a single data point, e.g. in a Bayesion Optimization (BO) setting
        Args:
            x: feature of shape
            y: float or of shape (1) or of shape (1,1)
        """
        x, y = _handle_point_input_dimensionality(x, y)
        xs, ys = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        self.add_data_points(xs, ys)

    @abstractmethod
    def predict(self, test_x, return_density=False, **kwargs):
        """Target predict. """
        pass

    def eval(self, test_x, test_y, **kwargs):
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_y: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """
        test_x, test_y = _handle_batch_input_dimensionality(test_x, test_y)

        # test_t_tensor = torch.from_numpy(test_y).contiguous().float().flatten().to(device)
        warnings.warn("find out if that is a diagonal distribution returned here")
        pred_dist = self.predict(test_x, return_density=True, *kwargs)
        avg_log_likelihood = pred_dist.log_prob(test_y) / test_y.shape[0]
        rmse = np.mean(np.pow(pred_dist.mean - test_y, 2)).sqrt()

        pred_dist_vect = self._vectorize_pred_dist(pred_dist)
        calibr_error = self._calib_error(pred_dist_vect, test_y)
        calibr_error_chi2 = _calib_error_chi2(pred_dist_vect, test_y)

        print("maybe this won't work, need to replace item", avg_log_likelihood)
        return avg_log_likelihood.item(), rmse.item(), calibr_error.item(), calibr_error_chi2

    def confidence_intervals(self, test_x, confidence=0.9, **kwargs):
        pred_dist = self.predict(test_x, return_density=True, **kwargs)
        pred_dist = self._vectorize_pred_dist(pred_dist)
        alpha = (1 - confidence) / 2
        ucb = pred_dist.icdf(np.ones(test_x.size) * (1 - alpha))
        lcb = pred_dist.icdf(np.ones(test_x.size) * alpha)
        return ucb, lcb

    """ ----- Private methods ----- """
    @abstractmethod
    def _recompute_posterior(self):
        """
        This method is called whenever new data is added. Subclasses should implement
        any computations that are needed prior to inference in this method and can assume that
        x_data and y_data are populated
        """
        pass

    def _clear_data(self):
        """
        Initializes the stored data to be empty.
        Notes:
            called both at initialisation, and when clearing the stored observations
        """
        self.xs_data = jnp.zeros((0, self.input_dim), dtype=np.double)
        self.ys_data = jnp.zeros((0,), dtype=np.double)
        self._num_train_points = 0

    def _set_normalization_stats(self, normalization_stats_dict=None):
        if normalization_stats_dict is None:
            self.x_mean, self.y_mean = np.zeros(self.input_dim), np.zeros(1)
            self.x_std, self.y_std = np.ones(self.input_dim), np.ones(1)
        else:
            self.x_mean = normalization_stats_dict['x_mean'].reshape((self.input_dim,))
            self.y_mean = normalization_stats_dict['y_mean'].squeeze()
            self.x_std = normalization_stats_dict['x_std'].reshape((self.input_dim,))
            self.y_std = normalization_stats_dict['y_std'].squeeze()

    def _calib_error(self, pred_dist_vectorized, test_t_tensor):
        raise NotImplementedError
        return _calib_error(pred_dist_vectorized, test_t_tensor)

    def _normalize_data(self, xs, ys=None):
        assert hasattr(self, "x_mean") and hasattr(self, "x_std"), "requires computing normalization stats beforehand"
        assert hasattr(self, "y_mean") and hasattr(self, "y_std"), "requires computing normalization stats beforehand"

        X_normalized = (xs - self.x_mean[None, :]) / self.x_std[None, :]

        if ys is None:
            return X_normalized
        else:
            Y_normalized = (ys - self.y_mean) / self.y_std
            return X_normalized, Y_normalized

    def _compute_normalization_stats(self, xs, ys):
        """
        Computes mean and std of the dataset
        Returns 
        """
        # save mean and variance of data for normalization
        if self.normalize_data:
            self.x_mean, self.y_mean = jnp.mean(X, axis=0), jnp.mean(Y, axis=0)
            self.x_std, self.y_std = jnp.std(X, axis=0) + 1e-8, jnp.std(Y, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = jnp.zeros(X.shape[1]), jnp.zeros(Y.shape[1])
            self.x_std, self.y_std = jnp.ones(X.shape[1]), jnp.ones(Y.shape[1])

    # def _vectorize_pred_dist(self, pred_dist: numpyro.distributions.Distribution) -> numpyro.distributions.Distribution:
    #     warnings.warn("check all use cases her")
    #     return numpyro.distributions.Normal(loc=pred_dist.mean, scale=jnp.sqrt(pred_dist.variance))


class RegressionModelMetaLearned(RegressionModel, ABC):
    """
    Abstracts the boilerplate functionality of a MetaLearnedRegression Model. This includes data normalization,
    fitting and inference. It also includes meta-functionality, in particular meta_fitting to multiple tasks,
    and meta_prediction on a new task.
    Public Methods:
        meta_fit: implementation of main algorithm, wi
        add_data_point(s): add data and refit the posterior
        predict: returns the predictive distribution for new points
    Notes:
        Subclasses need to implement the following:
            meta_fit(): Meta learning
            meta_predict(): target fitting and prediction
            _fit_posterior(): A method that is called after adding data points
            predict(xs): target prediction
    """
    @abstractmethod
    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        """Fits a hyper-posterior according to one of {F-,}PACOH-{MAP, SVGD}-{GP,NN}"""
        pass

    def meta_predict(self, context_x, context_y, test_x, return_density=False):
        """Convenience method that does a target_fit followed by a target_predict"""
        self._clear_data()
        self.add_data_points(context_x, context_y)
        self.predict(test_x, return_density=False)
        
    def eval_datasets(self, test_tuples, **kwargs):
        raise NotImplementedError
        """
        Performs meta-testing on multiple tasks / datasets.
        Computes the average test log likelihood, the rmse and the calibration error over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

        Returns: (avg_log_likelihood, rmse, calibr_error)

        """
        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        ll_list, rmse_list, calibr_err_list, calibr_err_chi2_list= list(zip(*[self.meta_eval(*test_data_tuple, **kwargs) for test_data_tuple in test_tuples]))

        return np.mean(ll_list), np.mean(rmse_list), np.mean(calibr_err_list), np.mean(calibr_err_chi2_list)

    def meta_eval(self,  context_x, context_y, test_x, test_y):
        raise NotImplementedError
        test_x, test_y = _handle_batch_input_dimensionality(test_x, test_y)
        test_y_tensor = torch.from_numpy(test_y).contiguous().float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.meta_predict(context_x, context_y, test_x, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_y_tensor) / test_y_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_y_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = self._calib_error(pred_dist_vect, test_y_tensor)
            calibr_error_chi2 = _calib_error_chi2(pred_dist_vect, test_y_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item(), calibr_error_chi2

    def _compute_meta_normalization_stats(self, meta_train_tuples):
        """
        Expects y to be flattened
        """
        xs_stack, ys_stack = map(list, zip(*[_handle_batch_input_dimensionality(x_train, y_train) for x_train, y_train in
                                             meta_train_tuples]))
        all_xs, all_ys = np.concatenate(xs_stack, axis=0), np.concatenate(ys_stack, axis=0)

        if self.normalize_data:
            self.x_mean, self.y_mean = np.mean(all_xs, axis=0), np.mean(all_ys, axis=0)
            self.x_std, self.y_std = np.std(all_xs, axis=0) + 1e-8, np.std(all_ys, axis=0) + 1e-8
        else:
            self.x_mean, self.y_mean = np.zeros(all_xs.shape[1]), np.zeros(1)
            self.x_std, self.y_std = np.ones(all_xs.shape[1]), np.ones(1)

    def _check_meta_data_shapes(self, meta_train_data):
        for i in range(len(meta_train_data)):
            meta_train_data[i] = _handle_batch_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]

        assert all([self.input_dim == train_x.shape[-1] and self.output_dim == train_t.shape[-1] for train_x, train_t in meta_train_data])

    def _prepare_data_per_task(self, x_data, y_data, flatten_y=True):
        # a) make arrays 2-dimensional
        x_data, y_data = _handle_batch_input_dimensionality(x_data, y_data, flatten_y)

        # b) normalize data
        x_data, y_data = self._normalize_data(x_data, y_data)

        if flatten_y and y_data.ndim == 2:
            assert y_data.shape[1] == 1
            y_data = y_data.flatten()

        # c) convert to tensors
        return x_data, y_data


    # def _vectorize_pred_dist(self, pred_dist: numpyro.distributions.Distribution):
    #     """
    #     Models the predictive distribution passed according to an independent, heteroscedastic Gaussian,
    #     i.e. forgets about covariance in case the distribution was multivariate.
    #     """
    #     return numpyro.distributions.Normal(pred_dist.mean, pred_dist.scale)

def _calib_error(pred_dist_vectorized, test_t_tensor):
    cdf_vals = pred_dist_vectorized.cdf(test_t_tensor)
    
    if test_t_tensor.shape[0] == 1:
        test_t_tensor = test_t_tensor.flatten()
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