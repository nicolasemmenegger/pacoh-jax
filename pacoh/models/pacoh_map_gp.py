from typing import Callable, Collection, Union, Optional

import haiku as hk
import jax.random
import numpy as np
import optax

from pacoh.models.meta_regression_base import RegressionModelMetaLearned
from pacoh.models.pure.pure_functions import construct_gp_base_learner
from pacoh.util.data_handling import DataNormalizer, normalize_predict
from pacoh.modules.means import JAXMean
from pacoh.modules.kernels import JAXKernel
from pacoh.util.initialization import initialize_model_with_state, initialize_optimizer


class PACOH_MAP_GP(RegressionModelMetaLearned):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_mode: str = "both",
        learn_likelihood: bool = True,
        weight_decay: float = 0.1,
        covar_module: Union[str, Callable[[], JAXKernel]] = "NN",
        mean_module: Union[str, Callable[[], JAXMean]] = "NN",
        mean_nn_layers: Collection[int] = (32, 32),
        kernel_nn_layers: Collection[int] = (32, 32),
        feature_dim: int = 2,
        task_batch_size: int = 5,
        num_tasks: int = None,
        lr: float = 1e-3,
        lr_decay: float = 1.0,
        normalize_data: bool = True,
        normalizer: Optional[DataNormalizer] = None,
        random_state: Optional[jax.random.PRNGKey] = None,
    ):
        """
        :param input_dim: The dimensionality of input points
        :param output_dim: The dimensionality of output points. Only output_dim = 1 is currently supported
        :param learning_mode: Can be one of "learn_mean", "learn_kernel", "both", or "vanilla_gp"
        :param learn_likelihood: Whether to learn the (homoscedastic) variance of the noise
        :param weight_decay: Weight decay on the parameters. Corresponds to setting a (certain)
            gaussian hyperprior on the parameters of the mean, kernel. We do not apply weight_decay
            on the likelihood parameters, since this coresponds to letting the noise go to softplus(0.0) = 0.69
        :param covar_module: Can be "NN", "SE" (Squared Exponential, i.e. RBF) or a Kernel object
        :param mean_module: Can be "NN", "constant", "zero", or a Mean
        :param mean_nn_layers: Size specifications of the hidden (!) layers used in the mean feature maps
        :param kernel_nn_layers: Size specifications of the hidden (!) layers used in the kernel feature maps
        :param feature_dim: In case covar_module is NN, feature_dim denotes the dimensionality of the output
            of the MLP that is then fed through a RBF kernel
        :param task_batch_size: The number of tasks in a batch
        :param num_tasks: The number of tasks we intend to train on. Required for jax.jit reasons
        :param lr: The learning rate of the AdamW optimizer
        :param lr_decay: The learning rate decay. 1.0 means no decay
        :param normalize_data: Whether to do everything with normalized data
        :param normalizer: Optional normalizer object. If none supplied, normalization stats are inferred from the
            training data
        :param random_state: A jax.random.PRNGKey to control all the pseudo-randomness inside this module

        Note: unlike in PACOH_SVGD_GP, we use the same gaussian hyperprior for all learnable parameters for simplicity
        (implicitly with weight decay). If you wish to use separate hyperpriors for the mean, kernel, and nn modules
        (used in mean and kernel of course), please use the SVGD class with 1 particle
        """
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
            likelihood_prior_mean=jax.nn.softplus(0.0),
            kernel_length_scale=jax.nn.softplus(0.0),
            kernel_output_scale=jax.nn.softplus(0.0),
        )
        init_fn, self._apply_fns = hk.multi_transform_with_state(pacoh_map_closure)
        self._rng, init_key = jax.random.split(self._rng)
        self.particle, empty_state = initialize_model_with_state(
            init_fn, init_key, (task_batch_size, input_dim)
        )
        self.state = self.empty_state = empty_state

        self._rng, sample_key = jax.random.split(self._rng)

        def target_post_prob(particle, meta_xs_batch, meta_ys_batch):
            """Assumes meta_xs is a list of len = task_batch_size, with each element being an array of a variable
            number of elements of shape [input_size]. Similarly for meta_ys
            """
            total_mll = 0.0
            for xs, ys in zip(meta_xs_batch, meta_ys_batch):
                mll, state = self._apply_fns.base_learner_mll_estimator(particle, empty_state, None, xs, ys)
                total_mll -= mll

            # log_prob assumes a batch of instances
            # We don't actually need that since we have weight decay, which is equivalent
            # reg = GaussianBelief.log_prob(
            #     self.hyperprior, jax.tree_map(lambda v: jnp.expand_dims(v, 0), particle)
            # )
            return total_mll  # + prior_weight * jnp.sum(reg)

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
    from pacoh.bo.meta_environment import RandomMixtureMetaEnv

    num_train_tasks = 20
    meta_env = RandomMixtureMetaEnv(random_state=np.random.RandomState(29))
    meta_train_data = meta_env.generate_uniform_meta_train_data(
        num_tasks=num_train_tasks, num_points_per_task=10
    )
    meta_test_data = meta_env.generate_uniform_meta_valid_data(
        num_tasks=50, num_points_context=10, num_points_test=160
    )

    NN_LAYERS = (32, 32)
    plot = True
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title("sample from the GP prior")
        plt.show()

    """ 2) Classical mean learning based on mll """

    print("\n ---- GPR mll meta-learning ---- ")

    # weight_decay = 0.01
    weight_decay = 0.0
    gp_model = PACOH_MAP_GP(
        input_dim=meta_env.domain.d,
        output_dim=1,
        num_tasks=num_train_tasks,
        weight_decay=weight_decay,
        task_batch_size=1,
        covar_module="SE",
        mean_module="constant",
        learning_mode="both",
        mean_nn_layers=NN_LAYERS,
        kernel_nn_layers=NN_LAYERS,
        learn_likelihood=True,
    )
    itrs = 0
    print("---- weight-decay =  %.4f ----" % weight_decay)
    for i in range(10):
        gp_model.meta_fit(
            meta_train_data, meta_valid_tuples=meta_test_data, log_period=1000, num_iter_fit=500
        )
        itrs += 500
        x_plot = np.linspace(meta_env.domain.l, meta_env.domain.u, num=150)
        x_context, t_context, x_test, y_test = meta_test_data[0]
        pred_mean, pred_std = gp_model.meta_predict(x_context, t_context, x_plot, return_density=False)
        ucb, lcb = (pred_mean + 2 * pred_std).flatten(), (pred_mean - 2 * pred_std).flatten()

        plt.scatter(x_test, y_test)
        plt.scatter(x_context, t_context)

        plt.plot(x_plot, pred_mean)
        plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2, color="orange")
        plt.title("GPR meta mll (weight-decay =  %.4f) itrs = %i" % (weight_decay, itrs))
        plt.show()

        print("AFTER ITERATION", itrs)
        print("first ds", gp_model.meta_eval(x_context, t_context, x_test, y_test))
        print("eval all datasets", gp_model.eval_datasets(meta_test_data))
