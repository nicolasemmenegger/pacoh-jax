from typing import Dict

import numpy as np
import jax
from jax import numpy as jnp
from abc import abstractmethod

from numpyro.distributions import MixtureSameFamily
from tqdm import trange

from pacoh.util.distributions import (
    diagonalize_gaussian,
    is_diagonal_gaussian_dist,
    get_diagonal_gaussian,
    is_gaussian_dist,
)
from pacoh.util.data_handling import (
    handle_batch_input_dimensionality,
    DataNormalizer,
    DataLoaderNumpy,
)
from pacoh.util.evaluation import calib_error, calib_error_chi2
from pacoh.util.abstract_attributes import AbstractAttributesABCMeta, abstractattribute


class RegressionModel(metaclass=AbstractAttributesABCMeta):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        flatten_ys: bool = False,
        random_state: jax.random.PRNGKey = None,
    ):
        """
        Abstracts the boilerplate functionality of a Regression Model. This includes data normalization,
        fitting and inference

        :param input_dim: The dimensionality of input points
        :param output_dim: The dimensionality of output points. Only output_dim = 1 is currently supported
        :param normalize_data: Whether to do everything with normalized data
        :param normalizer: Optional normalizer object. If none supplied, normalization stats are inferred from the
            training data
        :param flatten_ys: Whether to flatten the labels or not. Typically will be False when the (base)-learner
            is a NN and True if it is a GP, since there, only 1-dimensional inputs are allowed
        :param random_state: A jax.random.PRNGKey to control all the pseudo-randomness inside this module
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.flatten_ys = flatten_ys
        assert not flatten_ys or output_dim == 1, "implication flatten_ys => output_dim == 1 does not hold"
        self._xs_data = jnp.zeros((0, self._input_dim), dtype=np.double)
        if self.flatten_ys:
            self._ys_data = jnp.zeros((0,), dtype=np.double)
        else:
            self._ys_data = jnp.zeros((0, self._output_dim), dtype=np.double)
        self._num_train_points = 0

        self._rng = random_state if random_state is not None else jax.random.PRNGKey(42)
        self._normalizer = None
        if normalizer is None:
            self._normalizer = DataNormalizer(input_dim, output_dim, normalize_data, flatten_ys=flatten_ys)
        else:
            self._normalizer = normalizer
            assert normalizer.flatten_ys == flatten_ys, "instructions unclear, flatten_ys"
            assert normalizer.turn_off_normalization == (
                not normalize_data
            ), "instructions unclear, normalize"
            assert (
                output_dim == normalizer.output_dim and input_dim == normalizer.input_dim
            ), """Dimensions not matching"""

    def add_data_points(self, xs: np.ndarray, ys: np.ndarray, refit: bool = True):
        """
        Method to add some number of data points to the fitted regression model and refit.
        Args:
            xs: the points to add, of shape (num_data, self.input_dim) or (num_data) if input_dim == 1
            ys: the corresponding observations, of shape (num_data) or (num_data, 1)
            refit: whether to refit the posterior or not
        """
        xs, ys = self._normalizer.handle_data(xs, ys)

        # concatenate to new datapoints
        self._xs_data = np.concatenate([self._xs_data, xs])
        self._ys_data = np.concatenate([self._ys_data, ys])
        self._num_train_points += ys.shape[0]

        assert self._xs_data.shape[0] == self._ys_data.shape[0] == self._num_train_points

        if refit:
            self._recompute_posterior()

    def add_data_point(self, x, y, refit: bool = True):
        """
        Method to add a single data point, e.g. in a Bayesion Optimization (BO) setting
        :param x: feature of shape
        :param y: float or of shape (1) or of shape (1,1)
        :param refit: whether to refit the posterior or not
        """
        xs, ys = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        xs, ys = handle_batch_input_dimensionality(xs, ys)
        self.add_data_points(xs, ys, refit)

    def eval(self, test_xs: jnp.array, test_ys: jnp.array, pred_dist=None) -> Dict[str, float]:
        """
        Computes the average test log likelihood and the rmse on test data

        
        :param test_xs: (ndarray) test input data of shape (n_samples, ndim_x)
        :param test_ys: (ndarray) test target data of shape (n_samples, 1)
        :param pred_dist: numpyro.Distribution

        :returns a dictionary with keys 'avg. ll' 'rmse' 'calib err.' & 'calib err. chi2'

        """
        test_xs, test_ys = handle_batch_input_dimensionality(test_xs, test_ys, flatten_ys=self.flatten_ys)
        if pred_dist is None:
            pred_dist = self.predict(test_xs)

        avg_log_likelihood = pred_dist.log_prob(test_ys) / test_ys.shape[0]
        rmse = jnp.sqrt(jnp.mean(jax.lax.square(pred_dist.mean - test_ys)))
        calibr_error = calib_error(diagonalize_gaussian(pred_dist), test_ys)
        calibr_error_chi2 = calib_error_chi2(diagonalize_gaussian(pred_dist), test_ys)
        return {
            "avg. ll": avg_log_likelihood,
            "rmse": rmse,
            "calib err.": calibr_error,  # calibr_error.item(),
            "calib err. chi2": calibr_error_chi2,
        }

    def confidence_intervals(self, test_x, confidence=0.9):
        pred_dist = self.predict(test_x, return_density=True, return_full_covariance=False)
        alpha = (1 - confidence) / 2
        if not is_diagonal_gaussian_dist(pred_dist):
            if is_gaussian_dist(pred_dist):
                pred_dist = diagonalize_gaussian(pred_dist)
            elif isinstance(pred_dist, MixtureSameFamily):
                pred_dist = get_diagonal_gaussian(pred_dist.mean, jnp.sqrt(pred_dist.variance))
            else:
                raise NotImplementedError("I don't know how to approximate these confidence sets")
        ucb = pred_dist.base_dist.icdf((1 - alpha) * jnp.ones(pred_dist.event_shape))
        lcb = pred_dist.base_dist.icdf(alpha * jnp.ones(pred_dist.event_shape))
        return lcb, ucb

    @abstractmethod
    def predict(self, test_x, return_density=False, **kwargs):
        """Target predict."""
        pass

    def fit(self, xs_val=None, ys_val=None, log_period=500, num_iter_fit=None):
        """Default train loop, to be overwritten if custom behaviour is needed (e.g. for exact gp inference)."""
        dataloader = self._get_dataloader(
            self._xs_data, self._ys_data, self._batch_size, iterations=num_iter_fit
        )

        if xs_val is not None:
            assert ys_val is not None, "please specify both xs_val and ys_val"
            xs_val, ys_val = self._normalizer.handle_data(xs_val, ys_val)

        p_bar = trange(num_iter_fit)
        loss_list = []
        for i, batch in zip(p_bar, dataloader):
            xs_batch, ys_batch = batch
            curr_loss = self._step(xs_batch, ys_batch)
            loss_list.append(curr_loss)

            if i % log_period == 0:
                period_loss = jnp.mean(jnp.array(loss_list))
                loss_list = []
                message = dict(loss=period_loss)
                if xs_val is not None and ys_val is not None:
                    metric_dict = self.eval(xs_val, ys_val)
                    message.update(metric_dict)
                p_bar.set_postfix(message)

    """ ----- Private attributes ----- """

    @abstractattribute
    def _input_dim(self) -> int:
        ...

    @abstractattribute
    def _output_dim(self) -> int:
        ...

    @abstractattribute
    def _xs_data(self) -> jnp.array:
        ...

    @abstractattribute
    def _ys_data(self) -> jnp.array:
        ...

    @abstractattribute
    def _normalizer(self) -> DataNormalizer:
        ...

    @abstractattribute
    def _num_train_points(self) -> int:
        ...

    @abstractattribute
    def _output_dim(self) -> int:
        ...

    """ ----- Private methods ----- """

    @abstractmethod
    def _recompute_posterior(self):
        """
        This method is called whenever new data is added. Subclasses should implement
        any computations that are needed prior to inference in this method and can assume that
        x_data and y_data are populated
        """
        pass

    def _step(self, xs_batch, ys_batch):
        """One step of the train loop
        Note:
            this is not an abstract method because some modules may choose to not implement it.
        """
        raise NotImplementedError("This module does not implement the ._step method")

    def _clear_data(self):
        """
        Initializes the stored data to be empty.
        """
        self._xs_data = jnp.zeros((0, self._input_dim), dtype=np.double)
        if self.flatten_ys:
            self._ys_data = jnp.zeros((0,), dtype=np.double)
        else:
            self._ys_data = jnp.zeros((0, self._output_dim), dtype=np.double)
        self._num_train_points = 0
        self._recompute_posterior()

    def _get_dataloader(self, xs, ys, batch_size, iterations=2000) -> DataLoaderNumpy:
        """
        Returns an iterator to be used to sample minibatches from the dataset in the train loop
        :param xs, ys: the data set
        :param batch_size: num of points per iteration. -1 means no batching
        :param iterations: length of train loop with this dataloader
        :param batch_size: number of data point per task
        """
        self._rng, sampler_key = jax.random.split(self._rng)

        if batch_size == -1:
            batch_size = xs.shape[0]  # just use the whole dataset

        return DataLoaderNumpy(xs, ys, batch_size, iterations)
