import itertools
import unittest

import numpy as np

from pacoh.models.pacoh_map_gp import PACOH_MAP_GP


class ParameterChoice:
    def __init__(self, *args):
        self.choices = args

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.choices):
            raise StopIteration

        r = self.choices[self.i]
        self.i += 1
        return r


class SimpleIntegrationTest(unittest.TestCase):
    """
    This test just runs modules with enought configureations. Together with travis, this will allow us to
    specify maximal ranges for setup.py
    """
    def __init__(self, *args, **kwargs):
        super(SimpleIntegrationTest, self).__init__(*args, **kwargs)
        from pacoh.bo.meta_environment import RandomMixtureMetaEnv

        self.num_train_tasks = 20
        self.meta_env = RandomMixtureMetaEnv(random_state=np.random.RandomState(29))
        self.meta_train_data = self.meta_env.generate_uniform_meta_train_data(
            num_tasks=self.num_train_tasks, num_points_per_task=10
        )
        self.meta_test_data = self.meta_env.generate_uniform_meta_valid_data(
            num_tasks=50, num_points_context=10, num_points_test=160
        )

    def test_pacoh_map_gp(self):
        args = (1, 1) # input and output dim
        kwargs = {
            'learning_mode': ParameterChoice("both", "learn_mean", "learn_kernel"),
            'learn_likelihood': ParameterChoice(True, False),
            'covar_module': ParameterChoice("SE"),
            'mean_module': ParameterChoice("constant"),
            'weight_decay': 0.001,
            'mean_nn_layers': (32, 32),
            'kernel_nn_layers': (32, 32),
            'num_tasks': self.num_train_tasks
        }
        self._run_meta_module_with_configs(PACOH_MAP_GP, *args, **kwargs)
        kwargs = {
            'learning_mode': ParameterChoice("both"),
            'learn_likelihood': ParameterChoice(True, False),
            'covar_module': ParameterChoice("NN"),
            'mean_module': ParameterChoice("NN"),
            'weight_decay': 0.001,
            'mean_nn_layers': (32, 32),
            'kernel_nn_layers': (32, 32),
            'num_tasks': self.num_train_tasks
        }
        self._run_meta_module_with_configs(PACOH_MAP_GP, *args, **kwargs)



    def test_pacoh_svgd_gp(self):
        pass

    def test_fpacoh_map_gp(self):
        pass

    def test_vanilla_gp(self):
        pass

    def _run_meta_module_with_configs(self, mod_init, *args, **keyword_args):
        for config in parameter_product(keyword_args):
            self._run_meta_module_with_config(mod_init, *args, **config)

    def _run_meta_module_with_config(self, mod_init, *args, **keyword_args):
        meta_regression_module = mod_init(*args, **keyword_args)
        meta_regression_module.meta_fit(
            self.meta_train_data, meta_valid_tuples=self.meta_test_data, log_period=1000, num_iter_fit=10
        )

        xs_test = np.linspace(self.meta_env.domain.l, self.meta_env.domain.u, num=150)
        x_context, y_context, x_test, y_test = self.meta_test_data[0]
        pred_mean, pred_std = meta_regression_module.meta_predict(x_context, y_context, xs_test, return_density=False)
        self.assertFalse(np.isnan(np.sum(pred_mean + pred_std)))
        evals = meta_regression_module.meta_eval(x_context, y_context, x_test, y_test)
        for metric in evals.values():
            self.assertFalse(np.isnan(metric))


def parameter_product(range_dict):
    """ Takes a dictionary with some keys possibly being ParameterChoice objects. For those, it computes a cartesian product
    and returns all possible combination of dictionary arising from these Parameter choices. """
    keys = [k for k, v in range_dict.items() if isinstance(v, ParameterChoice)]
    vals = [v for _, v in range_dict.items() if isinstance(v, ParameterChoice)]
    items_without_ranges = [(k, v) for k,v in range_dict.items() if not isinstance(v, ParameterChoice)]

    prod = itertools.product(*vals)
    dicts = []
    for t in prod:
        items = list(zip(keys, t)) + items_without_ranges
        dicts.append(dict(items))

    return dicts



# if __name__ == "__main__":
#     import itertools
#
#     kwargs = {
#         'learning_mode': ParameterChoice("both", "mean", "kernel"),
#         'learn_likelihood': ParameterChoice(True, False),
#         'covar_module': ParameterChoice("SE", "NN"),
#         'weight_decay': 0.001,
#         'mean_nn_layers': (32, 32),
#         'kernel_nn_layers': (32, 32),
#         'num_task': 20
#     }
#     alt = {
#         'weight_decay': 0.001,
#         'mean_nn_layers': (32, 32),
#         'kernel_nn_layers': (32, 32),
#         'num_task': 20
#     }
#     print(len(parameter_product(alt)))
#     print(parameter_product(kwargs))
