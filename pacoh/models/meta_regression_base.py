from abc import ABC, abstractmethod

import jax
import numpy as np

import pacoh.util.evaluation
from pacoh.models.regression_base import RegressionModel
from pacoh.util.evaluation import calib_error_chi2
from pacoh.util.data_handling import handle_batch_input_dimensionality


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
        test_x, test_y = handle_batch_input_dimensionality(test_x, test_y)
        test_y_tensor = torch.from_numpy(test_y).contiguous().float().flatten().to(device)

        with torch.no_grad():
            pred_dist = self.meta_predict(context_x, context_y, test_x, return_density=True)
            avg_log_likelihood = pred_dist.log_prob(test_y_tensor) / test_y_tensor.shape[0]
            rmse = torch.mean(torch.pow(pred_dist.mean - test_y_tensor, 2)).sqrt()

            pred_dist_vect = self._vectorize_pred_dist(pred_dist)
            calibr_error = pacoh.util.evaluation.calib_error(pred_dist_vect, test_y_tensor)
            calibr_error_chi2 = calib_error_chi2(pred_dist_vect, test_y_tensor)

            return avg_log_likelihood.cpu().item(), rmse.cpu().item(), calibr_error.cpu().item(), calibr_error_chi2

    def _compute_meta_normalization_stats(self, meta_train_tuples):
        """
        Expects y to be flattened
        """
        xs_stack, ys_stack = map(list, zip(*[handle_batch_input_dimensionality(x_train, y_train) for x_train, y_train in
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
            meta_train_data[i] = handle_batch_input_dimensionality(*meta_train_data[i])
        self.input_dim = meta_train_data[0][0].shape[-1]
        self.output_dim = meta_train_data[0][1].shape[-1]

        assert all([self.input_dim == train_x.shape[-1] and self.output_dim == train_t.shape[-1] for train_x, train_t in meta_train_data])

    def _prepare_data_per_task(self, x_data, y_data, flatten_y=True):
        # a) make arrays 2-dimensional
        x_data, y_data = handle_batch_input_dimensionality(x_data, y_data, flatten_y)

        # b) normalize data
        x_data, y_data = self._normalize_data(x_data, y_data)

        if flatten_y and y_data.ndim == 2:
            assert y_data.shape[1] == 1
            y_data = y_data.flatten()

        # c) convert to tensors
        return x_data, y_data

    def _prepare_meta_train_tasks(self, meta_train_tuples, flatten_y=True):
        self._check_meta_data_shapes(meta_train_tuples)
        if self._normalization_stats is None:
            self._compute_meta_normalization_stats(meta_train_tuples)
        else:
            self._set_normalization_stats(self._normalization_stats)

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

    # def _vectorize_pred_dist(self, pred_dist: numpyro.distributions.Distribution):
    #     """
    #     Models the predictive distribution passed according to an independent, heteroscedastic Gaussian,
    #     i.e. forgets about covariance in case the distribution was multivariate.
    #     """
    #     return numpyro.distributions.Normal(pred_dist.mean, pred_dist.scale)