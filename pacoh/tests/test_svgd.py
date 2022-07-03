import sys
import unittest
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from pacoh.models.vanilla_bnn_svgd import BayesianNeuralNetworkSVGD
from pacoh.modules.kernels import pytree_rbf_set, rbf_cov, pytree_rbf, get_pytree_rbf_fn
from pacoh.tests.test_utils import get_simple_sinusoid_dataset
from pacoh.util.tree_util import pytree_unstack, broadcast_params
from pacoh.util.typing_util import Tree


class TreeTestCase(unittest.TestCase):
    @staticmethod
    def tree_assert_equal(first: Tree, second: Tree) -> None:
        jax.tree_map(lambda p, other: np.testing.assert_equal(p, other), first, second)

    @staticmethod
    def tree_assert_all_close(first: Tree, second: Tree) -> None:
        jax.tree_map(
            lambda p, other: np.testing.assert_allclose(p, other, rtol=1e-2),
            first,
            second,
        )


class TestSVGD(TreeTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # this is not the most canonical way to initialize a svgd object, but it avoids duplication of code
        self.kernel_bandwidth = 1000.0
        self.n = 10

        nn = BayesianNeuralNetworkSVGD(
            input_dim=1,
            output_dim=1,
            hidden_layer_sizes=(32, 32),
            prior_weight=0.001,
            bandwidth=self.kernel_bandwidth,
            learn_likelihood=True,
            n_particles=self.n,
        )

        print(nn)

        # construct a svgd object for the nn case
        self.svgd = nn.svgd
        self.xs, self.ys, _, __ = get_simple_sinusoid_dataset()
        self.particles = nn.particles

        list_of_trees = pytree_unstack(self.particles)
        list_of_params = []
        for tree in list_of_trees:
            flat_tree = jax.tree_flatten(tree)[0]  # discard treedef
            flat_params = list(map(lambda param: param.flatten(), flat_tree))
            all_params = jnp.concatenate(flat_params, axis=0)
            list_of_params.append(all_params)

        self.stacked_params = jnp.stack(list_of_params, axis=0)

    def test_phi_function(self):
        data = self.xs[:8], self.ys[:8]
        loss, test_update = self.svgd.neg_phi_update(self.particles, *data)

        # the above is a vmapped, and jitted, pytree compatible version
        # now compare this to a slower, but easier to understand version corresponding to the pseudocode in the paper

        def log_prob_distributed(
            particle, rng, *data
        ):  # kind of a hack because we only have access to the batched version...
            return self.svgd._log_prob_batched(broadcast_params(particle, self.n), rng, *data)

        unstacked_particles = pytree_unstack(self.particles)
        log_qs_fns = [
            jax.grad(lambda particle, *args: jnp.reshape(log_prob_distributed(particle, *args)[i], ()))
            for i in range(self.n)
        ]
        scores = [logq(unstacked_particles[i], None, *data) for i, logq in enumerate(log_qs_fns)]
        K = self.svgd.get_kernel_matrix(self.particles)
        updates = []
        kernel_grad_fns = [
            jax.grad(lambda tree: pytree_rbf(tree, unstacked_particles[i])) for i in range(self.n)
        ]
        for k in range(self.n):
            update_k = None
            for kp in range(self.n):
                kernel_val = K[kp, k]
                first_term = jax.tree_map(lambda p: p * kernel_val, scores[kp])
                second_term = kernel_grad_fns[k](unstacked_particles[kp])
                together = jax.tree_map(lambda p, o: -1.0 / self.n * (p + o), first_term, second_term)
                if update_k is None:
                    update_k = together
                else:
                    update_k = jax.tree_map(lambda a, b: a + b, update_k, together)

            updates.append(update_k)

        unstacked_updates = pytree_unstack(test_update)
        for k in range(self.n):
            self.tree_assert_all_close(updates[k], unstacked_updates[k])

    def test_pytree_rbf(self):
        """Tests whether the pytree and vmap based implementation of the kernel matrix for svgd is working."""
        pytree_res = get_pytree_rbf_fn(1.0, 1.0)(
            self.particles
        )  # pytree_rbf_set(self.particles, self.particles, 1.0, 1.0)
        n = self.stacked_params.shape[0]
        for i in range(n):
            for j in range(n):
                self.assertEqual(
                    pytree_res[i, j],
                    rbf_cov(
                        self.stacked_params[i],
                        self.stacked_params[j],
                        lambda: 1.0,
                        lambda: 1.0,
                    ),
                )
