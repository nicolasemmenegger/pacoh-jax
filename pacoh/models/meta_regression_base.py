import time
import warnings
from abc import abstractmethod
from typing import Optional, Union

import jax
from jax import numpy as jnp
from tqdm import trange
import numpy as np


from pacoh.models.regression_base import RegressionModel
from pacoh.util.data_handling import handle_batch_input_dimensionality, DataNormalizer, MetaDataLoaderTwoLevel, \
    MetaDataLoaderOneLevel
from pacoh.util.abstract_attributes import AbstractAttributesABCMeta, abstractattribute

class RegressionModelMetaLearned(RegressionModel, metaclass=AbstractAttributesABCMeta):
    """
    Abstracts the boilerplate functionality of a MetaLearnedRegression Model. This includes data normalization,
    fitting and inference. It also includes meta-functionality, in particular meta_fitting to multiple tasks,
    and meta_prediction on a new task.
    Public Methods:
        predict: predicts with a fitted base learner
        fit: fits the base learner
        meta_fit: meta-learns the prior
        add_data_point(s): add data and refit the posterior
        predict: returns the predictive distribution for new points
    Notes:
        Subclasses need to implement the following:
            meta_fit(): Meta learning
            meta_predict(): target fitting and prediction
            _fit_posterior(): A method that is called after adding data points
            predict(xs): target prediction
    """
    def __init__(self, input_dim: int, output_dim: int, normalize_data: bool = True,
                 normalizer: DataNormalizer = None, random_state: Optional[jax.random.PRNGKey] = None,
                 task_batch_size: int = -1, num_tasks: int = None, num_iter_meta_fit: int = -1, minibatch_at_dataset_level: bool = False,
                 dataset_batch_size: Optional[int] = None):
        super().__init__(input_dim, output_dim, normalize_data, normalizer, random_state)
        if num_tasks is None:
            raise ValueError("Please specify the number of tasks you intend to train on at initialisation time")

        if task_batch_size < 1:
            self._task_batch_size = num_tasks
        else:
            self._task_batch_size = min(task_batch_size, num_tasks)

        self._num_iter_meta_fit = num_iter_meta_fit
        self._minibatch_at_dataset_level = minibatch_at_dataset_level
        self._dataset_batch_size = dataset_batch_size
        if normalizer is None:
            self._provided_normaliser = None
            self._normalizer = None  # make sure we only set the normalizer based on the meta-tuples
        self.fitted = False

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, log_period=500, num_iter_fit=None):
        """
        :param meta_train_tuples:
        :param meta_valid_tuples:
        :param verbose:
        :param log_period:
        :param n_iter:
        :param minibatch_at_dataset_level: if this is true, we vectorize over the tasks as well.
            In this case the task batches need to be all of the same size, which can either be achieved by making all the
            meta-datasets the same size, or by minibatching both at the task AND dataset level (by default, we minibatch
            at the task level).
        """
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        if self._provided_normaliser is None:
            self._normalizer = DataNormalizer.from_meta_tuples(meta_train_tuples, True)
            # print("x_std:", self._normalizer.x_std)
            # print("y_std:", self._normalizer.y_std)
            # print("x_mean:", self._normalizer.x_mean)
            # print("y_mean:", self._normalizer.y_mean)
            self._provided_normaliser = self._normalizer


        self._check_meta_data_shapes(meta_train_tuples)
        num_iter_fit = self._num_iter_meta_fit if num_iter_fit is None else num_iter_fit
        meta_train_tuples = self._normalizer.handle_meta_tuples(meta_train_tuples)
        dataloader = self._get_meta_dataloader(meta_train_tuples,
                                               self._task_batch_size,
                                               iterations=num_iter_fit,
                                               return_array=self._minibatch_at_dataset_level,
                                               dataset_batch_size=self._dataset_batch_size)

        #if meta_valid_tuples is not None:
            # meta_valid_tuples = self._normalizer.handle_meta_tuples(meta_valid_tuples)

        p_bar = trange(num_iter_fit)
        avg_loss = 0.0
        message = {}
        for i, batch in zip(p_bar, dataloader):
            instantaneous_loss = self._meta_step(batch)
            itr = (i % log_period) + 1
            avg_loss = (itr-1)/itr*avg_loss + 1/itr*instantaneous_loss  # running average of loss over log_period
            message['loss'] = avg_loss
            p_bar.set_postfix(message)
            if i % log_period == 0:
                message = dict(loss=avg_loss)
                if meta_valid_tuples is not None:
                    agg_metric_dict = self.eval_datasets(meta_valid_tuples)
                    message.update(agg_metric_dict)
                p_bar.set_postfix(message)
                avg_loss = 0.0

        self.fitted = True

    def meta_predict(self, context_x, context_y, test_x, return_density=True):
        """Convenience method that does a target_fit followed by a target_predict"""
        self._clear_data()
        self.add_data_points(context_x, context_y, refit=True)
        return self.predict(test_x, return_density=return_density)

    def eval_datasets(self, test_tuples, **kwargs):
        """
        Performs meta-testing on multiple tasks / datasets.
        Computes the average test log likelihood, the rmse and the calibration error over multiple test datasets

        Args:
            test_tuples: list of test set tuples, i.e. [(test_context_x_1, test_context_y_1, test_x_1, test_y_1), ...]

        Returns: (avg_log_likelihood, rmse, calibr_error)

        """
        assert (all([len(valid_tuple) == 4 for valid_tuple in test_tuples]))

        results = [self.meta_eval(*test_data_tuple, **kwargs) for test_data_tuple in test_tuples]
        results_tup = [(res['avg. ll'], res['rmse'], res['calib err.'], res['calib err. chi2']) for res in results]
        ll, rmse, calib, chi2 = zip(*results_tup)

        return {
            'avg. ll': jnp.mean(jnp.array(ll)),
            'rmse': jnp.mean(jnp.array(rmse)),
            'calib err.': jnp.mean(jnp.array(calib)),
            'calib err. chi2': jnp.mean(jnp.array(chi2))
        }

    def meta_eval(self,  context_xs, context_ys, test_xs, test_ys):
        test_xs, test_ys = handle_batch_input_dimensionality(test_xs, test_ys)
        context_xs, context_ys = handle_batch_input_dimensionality(context_xs, context_ys)
        pred_dist = self.meta_predict(context_xs, context_ys, test_xs, return_density=True)
        return self.eval(test_xs, test_ys, pred_dist)

    """ ----- Mandatory attributes ----- """
    @abstractattribute
    def _task_batch_size(self) -> int:
        ...

    @abstractattribute
    def _dataset_batch_size(self) -> Optional[int]:
        ...

    @abstractattribute
    def _minibatch_at_dataset_level(self) -> bool:
        ...

    @abstractattribute
    def _num_iter_meta_fit(self) -> int:
        ...

    """ ----- Private methods ------ """
    def _check_meta_data_shapes(self, meta_train_data):
        """
        :param meta_train_data: List of tuples of meta training data
        :raises: AssertionError in case some dataset has wrong dimensions
        """
        for i in range(len(meta_train_data)):
            meta_train_data[i] = handle_batch_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]

        assert all([self.input_dim == train_x.shape[-1] and self.output_dim == train_t.shape[-1]
                    for train_x, train_t in meta_train_data])

    def _get_meta_dataloader(self,
                             meta_tuples,
                             task_batch_size,
                             iterations=2000,
                             dataset_batch_size=-1,
                             return_array=False) -> Union[MetaDataLoaderTwoLevel, MetaDataLoaderOneLevel]:
        """
        Returns an iterator to be used to sample minibatches from the dataset in the train loop
        :param meta_tuples: meta train set
        :param task_batch_size: num of train tasks per iterations. -1 means no batching
        :param iterations: length of train loop with this dataloader
        :param dataset_batch_size: number of data point per task
        :param return_array: whether the return tyype of the dataloader should be a three dimensional
        array (assumes batching or same size datasets, which could also be specified as -1).
        :return: a pytorch.DataLoader
        """
        self._rng, sampler_key = jax.random.split(self._rng)

        if task_batch_size == -1:
            task_batch_size = len(meta_tuples)  # just use the whole dataset
        elif task_batch_size > 0:
            pass
        else:
            raise AssertionError('task batch size must be either positive or -1')
        if return_array:
            if dataset_batch_size == -1 or dataset_batch_size is None:
                # check that they are all the same
                lengths = [[xs.shape[0], ys.shape[0]] for xs, ys in meta_tuples]
                lengths = np.array(lengths).flatten()
                assert np.all(lengths == lengths[0]), "If minibatching at task level without specifying a batch_size," \
                                                      "all datasets have to be of the same size"
                dataset_batch_size = lengths[0]
            elif dataset_batch_size > 0:
                pass
            else:
                raise AssertionError('batch size must be either positive or -1')

            return MetaDataLoaderTwoLevel(meta_tuples, task_batch_size, dataset_batch_size, iterations)
        else:
            return MetaDataLoaderOneLevel(meta_tuples, task_batch_size, iterations)

    @abstractmethod
    def _meta_step(self, mini_batch) -> float:
        """ should return some loss to display """
        pass
