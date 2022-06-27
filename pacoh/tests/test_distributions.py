from numpyro.distributions import Independent, Normal, MultivariateNormal
from jax import numpy as jnp
import jax

from pacoh.tests.test_svgd import TreeTestCase
from pacoh.util.distributions import vmap_dist


class TestVMAPDistribution(TreeTestCase):
    def get_dist(self, mean):
        return Independent(Normal(mean, jnp.ones_like(mean)), 2)

    def get_var(self, mean):
        return Independent(Normal(mean, jnp.ones_like(mean)), 2).variance

    def get_dist_and_var(self, mean):
        dist = self.get_dist(mean)
        return dist, dist.variance

    def get_nondiag_dist(self, mean):
        return MultivariateNormal(
            mean, jnp.diag(jnp.ones_like(mean)) + 0.001 * jnp.ones((mean.shape[0], mean.shape[0]))
        )

    def _get_data(self):
        key = jax.random.PRNGKey(42)
        mean = jax.random.normal(key, (8, 1))
        means = jax.random.normal(key, (3, 8, 1))
        twolevel = jax.random.normal(key, (4, 3, 8, 1))
        return key, mean, means, twolevel

    def test_output_dims(self):
        key, mean, means, twolevel = self._get_data()

        get_dists = vmap_dist(self.get_dist, out_axes=(0, 0))
        get_vars = vmap_dist(self.get_var)
        get_nondiag_dists = vmap_dist(self.get_nondiag_dist)

        simple_dist = self.get_dist(mean)  ##  .get_numpyro_distribution()
        batched_dist = get_dists(means)  ## .get_numpyro_distribution()
        batched_vars = get_vars(means)  ## .get_numpyro_distribution()
        multi_dist = self.get_nondiag_dist(mean.flatten())
        multi_dists = get_nondiag_dists(means.reshape((3, 8)))

        # Independent Normal
        self.assertEqual(simple_dist.batch_shape, ())
        self.assertEqual(simple_dist.event_shape, (8, 1))
        self.assertEqual(batched_dist.batch_shape, (3,))
        self.assertEqual(batched_dist.event_shape, (8, 1))

        # regular output
        self.assertEqual(batched_vars.shape, (3, 8, 1))

        # MultivariateNormal
        self.assertEqual(multi_dists.batch_shape, (3,))
        self.assertEqual(multi_dist.event_shape, (8,))

    def test_out_axes(self):
        key, mean, means, twolevel = self._get_data()
        multiple_outputs = vmap_dist(self.get_dist_and_var, out_axes=(0, 0))(means)
        different_axes = vmap_dist(self.get_dist_and_var, out_axes=(0, 1))(means)
        no_last_axis = vmap_dist(self.get_dist_and_var, out_axes=(0, None))(means)
        multiple_applications = vmap_dist(vmap_dist(self.get_dist))(twolevel)

        # Multiple outputs
        self.assertEqual(multiple_outputs[0].batch_shape, (3,))
        self.assertEqual(multiple_outputs[0].event_shape, (8, 1))
        self.assertEqual(multiple_outputs[1].shape, (3, 8, 1))

        # Different out_axes
        self.assertEqual(different_axes[0].batch_shape, (3,))
        self.assertEqual(different_axes[0].event_shape, (8, 1))
        self.assertEqual(different_axes[1].shape, (8, 3, 1))

        # one axis vmapped, one not
        self.assertEqual(no_last_axis[0].batch_shape, (3,))
        self.assertEqual(no_last_axis[0].event_shape, (8, 1))
        self.assertEqual(no_last_axis[1].shape, (8, 1))

        # apply vmap twice
        self.assertEqual(multiple_applications.batch_shape, (4, 3))
        self.assertEqual(multiple_applications.event_shape, (8, 1))
