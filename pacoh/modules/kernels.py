import functools

import haiku as hk
import jax

from pacoh.modules.common import PositiveParameter
from jax import numpy as jnp

from pacoh.util.tree_util import pytree_sum


def rbf_cov(x1, x2, ls_param, os_param):
    return os_param() * jnp.exp(-0.5 * (jnp.sum(jax.lax.square(x1 - x2))) / (ls_param() ** 2))


class JAXKernel(hk.Module):
    pass


class JAXRBFKernel(JAXKernel):
    def __init__(
        self,
        input_dim,
        length_scale=1.0,
        output_scale=1.0,
        log_ls_variance=0.0,  # 0.0 corresponds to deterministic initialization of the lengthscale
        log_os_variance=0.0,  # 0.0 corresponds to deterministic initialization of the outputscale
        length_scale_constraint_gt=0.0,
        output_scale_constraint_gt=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.output_scale = PositiveParameter(
            mean=output_scale,
            log_variance=log_os_variance,
            boundary_value=output_scale_constraint_gt,
            name="OutputScale",
        )
        self.length_scale = PositiveParameter(
            mean=length_scale,
            log_variance=log_ls_variance,
            boundary_value=length_scale_constraint_gt,
            name="LengthScale",
        )

    def __call__(self, x1, x2):
        return rbf_cov(x1, x2, self.length_scale, self.output_scale)


def pytree_sq_l2_dist(tree, other):
    func = lambda x1, x2: jnp.sum(jax.lax.square(x1 - x2))
    return pytree_sum(jax.tree_multimap(func, tree, other))


# Squared exponential kernel without anything learnable
def pytree_rbf(tree, other, lengthscale=1.0, outputscale=1.0):
    return outputscale * jnp.exp(-1.0 * pytree_sq_l2_dist(tree, other) / (lengthscale**2))


_pytree_rbf_mat_vec = jax.vmap(pytree_rbf, in_axes=(0, None, None, None))
_pytree_rbf_mat_mat = jax.vmap(_pytree_rbf_mat_vec, in_axes=(None, 0, None, None))


def get_pytree_rbf_fn(lengthscale, outputscale):
    return lambda p: _pytree_rbf_mat_mat(p, p, lengthscale, outputscale)


pytree_rbf_set = None


class JAXRBFKernelNN(JAXKernel):
    def __init__(
        self,
        input_dim,
        feature_dim,
        layer_sizes=(64, 64),
        length_scale=1.0,
        output_scale=1.0,
        length_scale_constraint_gt=0.0,
        output_scale_constraint_gt=0.0,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.output_scale = PositiveParameter(
            mean=output_scale,
            log_variance=0.0,
            boundary_value=output_scale_constraint_gt,
            name="OutputScale",
        )
        self.length_scale = PositiveParameter(
            mean=length_scale,
            log_variance=0.0,
            boundary_value=length_scale_constraint_gt,
            name="LengthScale",
        )

        self.nn_ftr_map = hk.nets.MLP(output_sizes=layer_sizes + (1,), activation=lambda x: x)

    def __call__(self, x1, x2=None):
        x1 = self.nn_ftr_map(x1)
        x2 = self.nn_ftr_map(x2)
        return rbf_cov(x1, x2, self.length_scale, self.output_scale)


if __name__ == "__main__":

    def forward(x1, x2):
        kernel = JAXRBFKernelNN(input_dim=1, feature_dim=1)
        return kernel(x1, x2)

    rng = jax.random.PRNGKey(42)
    rng, initkey = jax.random.split(rng)
    init, apply = hk.transform(forward)
    x1 = jnp.ones(1)
    x2 = jnp.zeros(1)
    params = init(rng, x1, x2)
    val, grad = jax.value_and_grad(apply)(params, rng, x1, x2)
    print(params)
    print(val)
    print(grad)
