import functools
import typing

import jax
import numpyro
from jax import numpy as jnp

from pacoh.modules.util import Tree, pytree_sum


class GaussianBeliefState(typing.NamedTuple):
    mean: Tree
    log_std: Tree

    @classmethod
    def initialize(cls, mean: float, std: float, template_tree=None):
        """
        :param initial_mean: a float indicating the mean for each parameter to use for initialization or a pytree
        :param initial_std: a float indicating the std for each parameter to use for initialization or a pytree
        :param template_tree: arbitrary pytree of the samples (pytree equivalent of event_dim). If none, the first two
        arguments must be the full pytrees
        """
        if template_tree is None:
            # we assume that initial_mean and initial_std are not floats
            mean = mean
            log_std = jax.tree_map(jnp.log, std)
        else:
            mean = jax.tree_map(lambda param: jnp.ones(param.shape) * mean, template_tree)
            log_std = jax.tree_map(lambda param: jnp.ones(param.shape) * jnp.log(std), template_tree)

        return cls(mean=mean, log_std=log_std)


    # def tree_flatten(self):
    #     flat_mean, mean_struct = jax.tree_util.tree_flatten(self.mean)
    #     flat_std, std_struct = jax.tree_util.tree_flatten(self.log_std)
    #     return (flat_mean, flat_std), (mean_struct, std_struct)
    #
    # @classmethod
    # def tree_unflatten(cls, aux, children):
    #     # children should be mean and log_std
    #     mean, log_std = children
    #
    #     return cls(mean, jax.tree_map(jnp.exp, log_std))

    @property
    def std(self):
        return jax.tree_map(jnp.exp, self.log_std)

    def copy(self):
        return self
        mean = jax.tree_map(jnp.copy, self.mean)
        std = jax.tree_map(jnp.copy, self.std)
        return GaussianBeliefState(mean, std)


class GaussianBelief:
    """A class that can be used both as a prior or as a posterior on some pytree of parameters."""

    @staticmethod
    def rsample(parameters: GaussianBeliefState, key: jax.random.PRNGKey, num_samples: int):
        """
        :param parameters: (mean, std) tree structure
        :param key: jax.random.PRNGKey
        :param: num_samples. The number of samples to draw for each leaf

        :return: A tree of the same shape as the GaussianBeliefState, except that each leaf gets expanded in the first
        dimension, which now corresponds to the number of samples
        """

        # get a tree of keys of the same shape as the mean and std tree, albeit not with same leaf shape (only need
        # one key to sample a Gaussian)
        num_params = len(jax.tree_util.tree_leaves(parameters.mean))
        keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(parameters.mean),
                                                 jax.random.split(key, num_params))

        def sample_leaf(leaf_mean, leaf_log_std_arr, key):
            leaf_mean = jnp.expand_dims(leaf_mean, -1)
            leaf_std_arr = jnp.exp(jnp.expand_dims(leaf_log_std_arr, -1))

            sample = jax.random.normal(key, (num_samples, *leaf_mean.shape), dtype=jnp.float32)
            sample = leaf_mean + sample * leaf_std_arr
            sample = sample.squeeze(axis=-1)
            return sample

        return jax.tree_multimap(sample_leaf, parameters.mean, parameters.log_std, keys_tree)

    @staticmethod
    def log_prob(parameters: GaussianBeliefState, samples: Tree):
        """
        Computes the log_probablity of a bunch of sampled parameters.
        Returns an array of size num_samples, each entry containing the log probablitiy of one sampled model
        """
        def leaf_log_prob(mean, std, sample):
            # assumes that params is a pytree of mean std and already stacked
            res = numpyro.distributions.Normal(loc=mean, scale=std).log_prob(sample) # (num_samples, *mean.shape)
            return jnp.sum(res, axis=tuple(range(1, res.ndim)))  # this sums up along the sample size direction

        sum = pytree_sum(jax.tree_multimap(leaf_log_prob, parameters.mean, parameters.std, samples))

        return sum