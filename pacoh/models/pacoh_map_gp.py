from typing import Callable, Collection, Union

import haiku as hk
import jax.random
import numpy as np
import optax
import torch
from jax import numpy as jnp

from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.models.pure.pure_functions import construct_gp_base_learner
from pacoh.modules.belief import GaussianBelief, GaussianBeliefState
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel
from pacoh.util.initialization import initialize_model_with_state, initialize_optimizer
from pacoh.util.tree import pytree_unstack


class PACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_mode: str = "both",
        learn_likelihood: bool = True,
        weight_decay: float = 0.0,
        feature_dim: int = 2,
        covar_module: Union[str, Callable[[], JAXKernel]] = "NN",
        mean_module: Union[str, Callable[[], JAXMean]] = "NN",
        mean_nn_layers: Collection[int] = (32, 32),
        kernel_nn_layers: Collection[int] = (32, 32),
        initial_noise_std: float = 0.01,
        task_batch_size: int = 5,
        num_tasks: int = None,
        lr: float = 1e-3,
        lr_decay: float = 1.0,
        prior_weight=1e-3,
        normalize_data: bool = True,
        normalizer: DataNormalizer = None,
        random_state: jax.random.PRNGKey = None,
    ):
        super().__init__(
            input_dim,
            output_dim,
            normalize_data,
            normalizer,
            random_state,
            task_batch_size,
            num_tasks,
            flatten_ys=True,
        )

        assert learning_mode in [
            "learn_mean",
            "learn_kernel",
            "both",
            "vanilla_gp",
        ], "Invalid learning mode"
        assert mean_module in ["NN", "constant", "zero"] or isinstance(
            mean_module, JAXMean
        ), "Invalid mean_module option"
        assert covar_module in ["NN", "SE"] or isinstance(
            covar_module, JAXKernel
        ), "Invalid covar_module option"
        """-------- Setup haiku differentiable functions and parameters -------"""
        pacoh_map_closure = construct_gp_base_learner(
            input_dim,
            output_dim,
            mean_module,
            covar_module,
            learning_mode,
            feature_dim,
            mean_nn_layers,
            kernel_nn_layers,
            learn_likelihood,
            initial_noise_std,
        )
        init_fn, self._apply_fns = hk.multi_transform_with_state(pacoh_map_closure)
        self._rng, init_key = jax.random.split(self._rng)
        self.particle, empty_state = initialize_model_with_state(
            init_fn, init_key, (task_batch_size, input_dim)
        )
        self.state = self.empty_state = empty_state

        def mean_std_map(*_):
            return 0.0, 1.0

        self.hyperprior = GaussianBeliefState.initialize_heterogenous(mean_std_map, self.particle)

        self._rng, sample_key = jax.random.split(self._rng)
        self.particle = pytree_unstack(GaussianBelief.rsample(self.hyperprior, sample_key, 1))[0]

        def target_post_prob(particle, meta_xs_batch, meta_ys_batch):
            """Assumes meta_xs is a list of len = task_batch_size, with each element being an array of a variable
            number of elements of shape [input_size]. Similarly for meta_ys
            """
            total_mll = 0.0
            for xs, ys in zip(meta_xs_batch, meta_ys_batch):
                mll, state = self._apply_fns.base_learner_mll_estimator(particle, empty_state, None, xs, ys)
                total_mll -= mll

            # log_prob assumes a batch of instances
            reg = GaussianBelief.log_prob(
                self.hyperprior, jax.tree_map(lambda v: jnp.expand_dims(v, 0), particle)
            )
            return total_mll - prior_weight * jnp.sum(reg)

        def target_log_prob_array(particle, meta_xs, meta_ys):
            """meta_{xs, ys} are both three dimensional jnp.arrays -> which enables vmapping the train loop over tasks"""
            pass

        self.target_val_and_grad = jax.jit(jax.value_and_grad(target_post_prob))

        self.optimizer, self.optimizer_state = initialize_optimizer(
            "AdamW", lr, self.particle, lr_decay, weight_decay=weight_decay
        )

    def _meta_step(self, mini_batch) -> float:
        xs_batch, ys_batch = mini_batch
        loss, grad = self.target_val_and_grad(self.particle, xs_batch, ys_batch)
        updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state, self.particle)
        self.particle = optax.apply_updates(self.particle, updates)
        return loss

    @normalize_predict
    def predict(self, xs):
        pred_dist, _ = self._apply_fns.base_learner_predict(self.particle, self.state, None, xs)
        return pred_dist

    def _recompute_posterior(self):
        # use the stored data in xs_data, ys_data to instantiate a base_learner
        _, self.state = self._apply_fns.base_learner_fit(
            self.particle, self.empty_state, None, self._xs_data, self._ys_data
        )


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_debug_nans", False)
    config.update("jax_disable_jit", False)

    from experiments.data_sim import SinusoidDataset

    data_sim = SinusoidDataset(random_state=np.random.RandomState(29))
    num_train_tasks = 20
    meta_train_data = data_sim.generate_meta_train_data(n_tasks=num_train_tasks, n_samples=10)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=50, n_samples_context=10, n_samples_test=160)

    NN_LAYERS = (32, 32, 32, 32)

    plot = False
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title("sample from the GP prior")
        plt.show()
    """ 2) Classical mean learning based on mll """

    print("\n ---- GPR mll meta-learning ---- ")

    torch.set_num_threads(2)

    for weight_decay in [0.5]:
        pacoh_map = PACOH_MAP_GP(
            1,
            1,
            learning_mode="both",
            weight_decay=weight_decay,
            task_batch_size=5,
            num_tasks=num_train_tasks,
            covar_module="NN",
            mean_module="NN",
            mean_nn_layers=NN_LAYERS,
            feature_dim=2,
            kernel_nn_layers=NN_LAYERS,
        )

        itrs = 0
        print("---- weight-decay =  %.4f ----" % weight_decay)

        for i in range(40):
            n_iter = 50
            pacoh_map.meta_fit(meta_train_data, meta_test_data, log_period=1000, num_iter_fit=n_iter)

            itrs += n_iter

            x_plot = np.linspace(-5, 5, num=150)
            x_context, y_context, x_test, y_test = meta_test_data[0]
            pred_dist = pacoh_map.meta_predict(x_context, y_context, x_plot, return_density=True)
            pred_mean = pred_dist.mean

            plt.scatter(x_test, y_test, color="green")  # the unknown target test points
            plt.scatter(x_context, y_context, color="red")  # the target train points
            plt.plot(x_plot, pred_mean)  # the curve we fitted based on the target test points

            lcb, ucb = pacoh_map.confidence_intervals(x_plot)
            plt.fill_between(x_plot, lcb.flatten(), ucb.flatten(), alpha=0.2, color="green")
            plt.title("GPR meta mll (weight-decay =  %.4f) itrs = %i" % (weight_decay, itrs))
            plt.show()
