import warnings
from functools import partial

import jax
import optax

from jax import numpy as jnp

from pacoh.modules.kernels import pytree_sq_l2_dist, pytree_rbf, pytree_rbf_set, get_pytree_rbf_fn
from pacoh.util.tree import Tree, pytree_sum, pytree_shape, pytree_unstack


class SVGD:
    def __init__(self, target_log_prob_batched, kernel_forward_pytree, optimizer, optimizer_state):
        """
        :param target_log_prob: pure forward function that takes [(tree with vectorized leaves of particle params), possibly rngs, shared data batch]
        :param kernel_forward: The vmapped version of a function which takes two pytrees of parameters and thus returns a kernel matrix
        :param optimizer: An optimizer working one the pytree of particles

        Notes:
            * in the standard case, target_log_probs will take model params and output an MLL + prior ll estimate
            * in the meta-learning case, target_log_probs will take prior parameters and return an expected MLL + prior ll estimate
        """
        # a) score function
        self.target_log_prob_batched = target_log_prob_batched
        self.score = jax.grad(lambda params, rng, *data: jnp.sum(target_log_prob_batched(params, rng, *data)))

        # b) kernel grad and matrix function
        def sum_cols_of_kernel(params):
            # note that both data and rng are not needed
            # if a leaf has shape (*p), then params has (n, params)
            K = kernel_forward_pytree(params)
            return jnp.sum(K, axis=1)

        self.sum_of_cols = sum_cols_of_kernel

        self.kernel_grads = jax.jacrev(sum_cols_of_kernel)
        self.get_kernel_matrix = kernel_forward_pytree

        # c) setup optimizer
        self.optimizer = optimizer
        self.optimizer_state = optimizer_state

    @partial(jax.jit, static_argnums=0)
    def neg_phi_update(self, particles: Tree, *data):
        """
        :param particles: A pytree of particles, where there are num_particles in the first dimension of the leaves
        :param data: A batch of data to evaluate the likelihood on
        :return: A pytree of the same shape as particles, corresponding to the update applied to it (can be interpreted
        as a function space gradient)
        """
        warnings.warn("some sort of rng will be needed in the meta-learning case")
        score_val = self.score(particles, None, *data)  # shape (n, *p)
        # (j, *ids) corresponds to grad_{x_j} p(x_j)

        kernel_mat_val = self.get_kernel_matrix(particles)  # shape (n,n)
        n_particles = kernel_mat_val.shape[0]
        # (i, j) corresponds to K(x_j,x_i)

        kernel_grads_val = self.kernel_grads(particles) # shape (n, n, *p)
        # (i, j, *ids) corresponds to (grad_{x_j} f(x_j, x_i))[ids]

        @jax.jit
        def neg_phi_update_leaf(leaf_score, leaf_kernel_grads):
            # kernel_mat_val has shape (n, n) and leaf_score has shape (n, *p)
            # the resulting product has shape (n, *p) as well
            res = (jnp.tensordot(kernel_mat_val, leaf_score, axes=1) + jnp.sum(leaf_kernel_grads, axis=1)) / n_particles
            return -res

        result = jax.tree_multimap(neg_phi_update_leaf, score_val, kernel_grads_val)  # kernel_mat_val is symmetric
        return result

    def step(self, particles, *data):
        decrease = self.neg_phi_update(particles, *data)
        updates, self.optimizer_state = self.optimizer.update(decrease, self.optimizer_state, particles)
        particles = optax.apply_updates(particles, updates)
        return particles


if __name__ == "__main__":
    params = {'w1': jnp.array([[0.1, 0.1], [0.2, 0.1]]), 'b1': jnp.array([[3.0], [-1.0]])}
    p1 = {'w1': jnp.array([0.1, 0.1]), 'w2': jnp.array([3.0])}
    p2 = {'w1': jnp.array([0.2, 0.1]), 'w2': jnp.array([-1.0])}

    print(jax.tree_util.tree_leaves(params))

    def k_vec_vec(param1, param2):
        # the linear kernel
        return jnp.exp(pytree_sum(jax.tree_multimap(lambda v1, v2: jnp.sum(-(v1-v2)**2), param1, param2)))

    print("no vmap", k_vec_vec(p1, p2))
    k_mat_vec = jax.vmap(k_vec_vec, in_axes=(0, None))
    k_mat_mat = jax.vmap(k_mat_vec, in_axes=(None, 0))

    def F(params): # the function that sums up the kernel matrix elements along a direction
        return jnp.sum(k_mat_mat(params, params), axis=1)

    kernel_gradients = (jax.jacfwd(F))(params)
    print(jax.tree_map(lambda p: p.shape, kernel_gradients))

    # test to know axes
    def f(x):
        return jnp.array([x[0][0]**2 + x[1][0]**2, x[1][0]**2, x[0][0]**2])

    x = jnp.array([[1.0], [2.0]])
    grads = jax.jacfwd(f)(x)
    print(f(x))
    print(grads.shape)

