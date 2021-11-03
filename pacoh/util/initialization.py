from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
import haiku as hk

from pacoh.util.tree import pytree_unstack


def _call_init(init: Callable[..., hk.Params],
               prng_key: jax.random.PRNGKey,
               *shapes: Tuple[int, ...]) -> Union[hk.Params, Tuple[hk.Params, hk.State]]:
    """ Creates parameters for a given model initializer by creating some dummy data to feed through the given init.
        Depending on whether the underlying hk.Module uses state, it returns just parameters, or both parameters and
        an initial state.
    """
    dummy_data = []
    for shape in shapes:
        dummy_data.append(jnp.ones(shape))
    return init(prng_key, *dummy_data)


""" Public aliases that make it clear to the user what the return type is"""
initialize_model = _call_init  # returns parameters
initialize_model_with_state = _call_init  # returns parameters and state


def _call_init_batched(init: Callable[..., hk.Params],
                       n_models: int,
                       prng_key: jax.random.PRNGKey,
                       *shapes: Tuple[int, ...]):
    """ Creates parameters for a given batched (vmapped) model initializer by creating some dummy data to feed through
        init. Depending on whether the underlying hk.Module uses state, it returns parameters or a tuple of parameters
        and initial states. Shapes refers to the data that must be supplied to the init function.
    """
    dummy_data = []
    for shape in shapes:
        dummy_data.append(jnp.ones(shape))

    prng_keys = jax.random.split(prng_key, n_models)
    return init(prng_keys, *dummy_data)


def initialize_batched_model(init: Callable[..., hk.Params],
                             n_models: int,
                             prng_key: jax.random.PRNGKey,
                             *shapes: Tuple[int, ...]) -> Tuple[hk.Params, hk.Params]:
    """ Initializes n_models according to the supplied vmapped init function.
        The second return value is a tree of parameters for a single module, which is useful for specifying
        distributions over parameters.
    """
    params = _call_init_batched(init, n_models, prng_key, *shapes)
    param_template = pytree_unstack(params)[0]
    return params, param_template


def initialize_batched_model_with_state(init, n_models, prng_key, *shapes) -> Tuple[hk.Params, hk.Params, hk.State]:
    """ Initializes n_models stateful models according to the supplied vmapped init function """
    params, states = _call_init_batched(init, n_models, prng_key, *shapes)
    param_template = pytree_unstack(params)[0]
    return params, param_template, states
