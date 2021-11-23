import time
import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import jax
import haiku as hk
from jax import numpy as jnp
from tqdm import trange

from pacoh.models.regression_base import RegressionModel
from pacoh.util.data_handling import handle_batch_input_dimensionality, DataNormalizer


class TaskDict(NamedTuple):
    xs: jnp.array
    ys: jnp.array
    state: Optional[hk.State]

class RegressionModelMetaLearned(RegressionModel, ABC):
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
                 normalizer: DataNormalizer = None, random_state: jax.random.PRNGKey = None):
        super().__init__(input_dim, output_dim, normalize_data, normalizer, random_state)
        if normalizer is None:
            self._normalizer = None # make sure we only set the normalizer based on the meta-tuples
        self.fitted = False

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, log_period=500, num_iter_fit=None, vectorize_over_tasks=True):
        """
        :param meta_train_tuples:
        :param meta_valid_tuples:
        :param verbose:
        :param log_period:
        :param n_iter:
        :param vectorize_over_tasks: if this is true, we vectorize over the tasks as well.
            In this case the task batches need to be all of the same size, which can either be achieved by making all the
            meta-datasets the same size, or by minibatching both at the task AND dataset level (by default, we minibatch
            at the task level).
        """
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)
        meta_batch_sampler = self._get_meta_batch_sampler(task_dicts, self.task_batch_size, self.data_set_batch_size, vectorize=vectorize_over_tasks)

        t = time.time()
        loss_list = []
        num_iter_fit = self.num_iter_fit if num_iter_fit is None else num_iter_fit # TODO do I really need this? seems wrong to specify loop size at initialisation
        pbar = trange(num_iter_fit)

        for i in pbar:
            # a) choose a minibatch.
            # train_batch_sampler returns one of the following:
            # - 2 lists of size task_batch_size of xs and ys each of variable sizes
            # - an array of size (task_batch_size, data_set_batch_size, {input_dim or output_dim resp.})
            xs_task_list, ys_task_list = next(meta_batch_sampler)
            loss = self._meta_step(xs_task_list, ys_task_list, vectorize_over_tasks)
            loss_list.append(loss)

            if i % log_period == 0:
                loss = jnp.mean(jnp.array(loss_list))
                loss_list = []
                message = dict(loss=loss)
                if meta_valid_tuples is not None:
                    agg_metric_dict = self.eval_datasets(meta_valid_tuples)
                    message.update(agg_metric_dict)
                pbar.set_postfix(message)

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

        ll, rmse, calib, chi2 = list(zip(*[self.meta_eval(*test_data_tuple, **kwargs)
                                           for test_data_tuple in test_tuples]))

        return {
            'avg. ll': jnp.mean(ll),
            'rmse': jnp.mean(rmse),
            'calib err.': jnp.mean(calib),
            'calib err. chi2': jnp.mean(chi2)
        }

    def meta_eval(self,  context_xs, context_ys, test_xs, test_ys):
        test_xs, test_ys = handle_batch_input_dimensionality(test_xs, test_ys)
        context_xs, context_ys = handle_batch_input_dimensionality(context_xs, context_ys)
        pred_dist = self.meta_predict(context_xs, context_ys, test_xs, return_density=True)
        return self.eval(test_xs, test_ys, pred_dist)

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

    def _prepare_meta_train_tasks(self, meta_train_tuples):
        raise NotImplementedError
        """
        Sets up the model for meta fitting with the meta train tuples given
        If no normalisation stats are previously set, computes them from the meta data sets
        """
        self._check_meta_data_shapes(meta_train_tuples)
        if self._normalizer is None:
            self._normalizer = DataNormalizer.from_meta_tuples(meta_train_tuples)
        else:
            warnings.warn("I removed the else branch at some point, this should already have been set")

        if not self.fitted:
            self._rng, init_rng = jax.random.split(self._rng) # random numbers
            self.params, self.empty_state = self._init_fn(init_rng, meta_train_tuples[0][0]) # prior parameters, initial state
            self.opt_state = self.optimizer.init(self.params)  # optimizer on the prior params

        task_dicts = []

        for xs,ys in meta_train_tuples:
            # state of a gp can encapsulate caches of already fitted data and hopefully speed up inference
            _, state = self._apply_fns.base_learner_fit(self.params, self.empty_state, self._rng, xs, ys)
            task_dict = {
                'xs_train': xs,
                'ys_train': ys,
                'hk_state': state
            }

            if flatten_y:
                task_dict['ys_train'] = ys.flatten()

            task_dicts.append(task_dict)

        return task_dicts

    @abstractmethod
    def _meta_step(self, xs_tasks, ys_tasks, vectorize_over_tasks):
        pass
