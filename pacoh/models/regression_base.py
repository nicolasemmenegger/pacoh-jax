import sys
from typing import Dict

import numpy as np
import jax
from jax import numpy as jnp
from abc import ABC, abstractmethod

from tqdm import trange

from pacoh.modules.distributions import diagonalize_gaussian
from pacoh.util.data_handling import handle_batch_input_dimensionality, Sampler, DataNormalizer
from pacoh.util.evaluation import calib_error, calib_error_chi2
from pacoh.util.abstract_attributes import AbstractAttributesABCMeta, abstractattribute


class RegressionModel(metaclass=AbstractAttributesABCMeta):
    def __init__(self, input_dim: int, output_dim: int, normalize_data: bool = True,
                 normalizer: DataNormalizer = None, flatten_ys: bool = False, random_state: jax.random.PRNGKey = None):
        """
        Abstracts the boilerplate functionality of a Regression Model. This includes data normalization,
        fitting and inference
        Public Methods:
            add_data_point(s): add data and refit the posterior
            predict: returns the predictive distribution for new points
        Args:
            input_dim: The dimensions of the input features
            output_dim: The output dimension of the learner
            normalize_data: Whether to do data normalisation prior to fitting and inference
            normalization_stats: Normalization stats to set and use. If none, the default is to set it so
                nothing happens
            random_state: The master PRNGKey to be used in downstream modules
        Notes:
            Subclasses need to implement the following:
                _recompute_posterior(): A method that is called after adding data points
                predict(xs): target prediction
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.flatten_ys = flatten_ys
        assert not flatten_ys or output_dim == 1, "implication flatten_ys => output_dim == 1 does not hold"
        self._clear_data()

        self._rng = random_state if random_state is not None else jax.random.PRNGKey(42)
        self._normalizer = None
        if normalizer is None:
            self._normalizer = DataNormalizer(input_dim, output_dim, normalize_data, flatten_ys=flatten_ys)
        else:
            self._normalizer = normalizer
            assert normalizer.flatten_ys == flatten_ys, "instructions unclear, flatten_ys"
            assert normalizer.turn_off_normalization == (not normalize_data), "instructions unclear, normalize"
            assert output_dim == normalizer.output_dim and input_dim == normalizer.input_dim, """Dimensions not matching"""

    def add_data_points(self, xs: np.ndarray, ys: np.ndarray, refit: bool = True):
        """
        Method to add some number of data points to the fitted regression model and refit.
        Args:
            xs: the points to add, of shape (num_data, self.input_dim) or (num_data) if input_dim == 1
            ys: the corresponding observations, of shape (num_data) or (num_data, 1)
            refit: whether to refit the posterior or not
        """
        # xs, ys = handle_batch_input_dimensionality(xs, ys, flatten_ys=False)
        # assert xs.ndim == 2 and xs.shape[1] == self._input_dim, "Something is wrong with your data"
        # assert ys.ndim == 2 and ys.shape[1] == self._output_dim, "Something is wrong with your data"
        # handle input dimensionality and normalize data
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
        Args:
            x: feature of shape
            y: float or of shape (1) or of shape (1,1)
            refit: whether to refit the posterior or not
        """
        xs, ys = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
        xs, ys = handle_batch_input_dimensionality(xs, ys)
        self.add_data_points(xs, ys, refit)

    def eval(self, test_xs: jnp.array, test_ys: jnp.array, pred_dist=None) -> Dict[str, float]:
        """
        Computes the average test log likelihood and the rmse on test data

        Args:
            test_x: (ndarray) test input data of shape (n_samples, ndim_x)
            test_y: (ndarray) test target data of shape (n_samples, 1)

        Returns: (avg_log_likelihood, rmse)

        """
        test_xs, test_ys = handle_batch_input_dimensionality(test_xs, test_ys, flatten_ys=self.flatten_ys)
        if pred_dist is None:
            pred_dist = self.predict(test_xs)

        test = pred_dist.log_prob(test_ys) # this should just be a scalar no
        avg_log_likelihood = pred_dist.log_prob(test_ys) / test_ys.shape[0]
        rmse = jnp.sqrt(jnp.mean(jax.lax.square(pred_dist.mean - test_ys)))
        calibr_error = calib_error(diagonalize_gaussian(pred_dist), test_ys)
        calibr_error_chi2 = calib_error_chi2(diagonalize_gaussian(pred_dist), test_ys)
        return {
            'avg. ll': avg_log_likelihood.item(),
            'rmse': rmse.item(),
            'calib err.': calibr_error.item(),
            'calib err. chi2': calibr_error_chi2
        }

    def confidence_intervals(self, test_x, confidence=0.9):
        pred_dist = self.predict(test_x, return_density=True, return_full_covariance=False)
        alpha = (1 - confidence) / 2
        ucb = pred_dist.base_dist.icdf((1 - alpha)*jnp.ones(pred_dist.event_shape))
        lcb = pred_dist.base_dist.icdf(alpha*jnp.ones(pred_dist.event_shape))
        return lcb, ucb

    @abstractmethod
    def predict(self, test_x, return_density=False, **kwargs):
        """Target predict. """
        pass

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        """ Default train loop, to be overwritten if custom behaviour is needed (e.g. for exact gp inference). """
        train_batch_sampler = self._get_batch_sampler(self._xs_data, self._ys_data, self._batch_size)
        loss_list = []
        pbar = trange(num_iter_fit)
        for i in pbar:
            x_batch, y_batch = next(train_batch_sampler)
            loss = self._step(x_batch, y_batch)
            loss_list.append(loss)

            if i % log_period == 0:
                loss = jnp.mean(jnp.array(loss_list))
                loss_list = []
                message = dict(loss=loss)
                if x_val is not None and y_val is not None:
                    metric_dict = self.eval(x_val, y_val)
                    message.update(metric_dict)
                pbar.set_postfix(message)

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
        """ One step of the train loop
            Note:
                this is not an abstract method because some modules may choose to not implement it.
        """
        raise NotImplementedError("This module does not implement the ._step method")

    def _clear_data(self):
        """
        Initializes the stored data to be empty.
        Notes:
            called both at initialisation, and when clearing the stored observations
        """
        self._xs_data = jnp.zeros((0, self._input_dim), dtype=np.double)
        if self.flatten_ys:
            self._ys_data = jnp.zeros((0,), dtype=np.double)
        else:
            self._ys_data = jnp.zeros((0, self._output_dim), dtype=np.double)
        self._num_train_points = 0

    def _get_batch_sampler(self, xs, ys, batch_size, shuffle=True):
        """
        Returns an iterator to be used to sample minibatches from the dataset in the train loop
        :param xs: The feature vectors
        :param ys: The labels
        :param batch_size: The size of the batches. If -1, will be the whole data set
        :param shuffle: If this is true, we initially shuffle the data, and then iterate
                        If it is false, we subsample each time the iterator gets queried
        :return: a Sampler object (which is itself an iterator)
        """
        # iterator that shuffles and repeats the data
        xs, ys = handle_batch_input_dimensionality(xs, ys)
        num_train_points = xs.shape[0]

        if batch_size == -1:
            batch_size = num_train_points
        elif batch_size > 0:
            pass
        else:
            raise AssertionError('batch size must be either positive or -1')

        self._rng, sampler_key = jax.random.split(self._rng)
        return Sampler(xs, ys, batch_size, sampler_key, shuffle)

