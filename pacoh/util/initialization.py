import functools
import warnings
from typing import Callable, Tuple, List, Union

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from pacoh.util.tree import pytree_unstack


def _call_init(
    init: Callable[..., hk.Params], prng_key: jax.random.PRNGKey, *shapes: Tuple[int, ...]
) -> Union[hk.Params, Tuple[hk.Params, hk.State]]:
    """Creates parameters for a given model initializer by creating some dummy data to feed through the given init.
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


def _call_init_batched(
    init: Callable[..., hk.Params], n_models: int, prng_key: jax.random.PRNGKey, *shapes: Tuple[int, ...]
):
    """Creates parameters for a given batched (vmapped) model initializer by creating some dummy data to feed through
    init. Depending on whether the underlying hk.Module uses state, it returns parameters or a tuple of parameters
    and initial states. Shapes refers to the data that must be supplied to the init function.
    """
    dummy_data = []
    for shape in shapes:
        dummy_data.append(jnp.ones(shape))

    prng_keys = jax.random.split(prng_key, n_models)
    return init(prng_keys, *dummy_data)


def initialize_batched_model(
    init: Callable[..., hk.Params], n_models: int, prng_key: jax.random.PRNGKey, *shapes: Tuple[int, ...]
) -> Tuple[hk.Params, hk.Params]:
    """Initializes n_models according to the supplied vmapped init function.
    The second return value is a tree of parameters for a single module, which is useful for specifying
    distributions over parameters.
    """
    params = _call_init_batched(init, n_models, prng_key, *shapes)
    param_template = pytree_unstack(params)[0]
    return params, param_template


def initialize_batched_model_with_state(
    init, n_models, prng_key, *shapes
) -> Tuple[hk.Params, hk.Params, hk.State]:
    """Initializes n_models stateful models according to the supplied vmapped init function"""
    params, states = _call_init_batched(init, n_models, prng_key, *shapes)
    param_template = pytree_unstack(params)[0]
    return params, param_template, states


def initialize_optimizer(optimizer, lr, parameters, lr_decay=None, mask_fn=None, weight_decay=None):
    # check options
    if mask_fn is not None:
        assert optimizer == "AdamW", "mask not supported for any other option than ADAMW"

    if weight_decay is not None and optimizer != "AdamW":
        warnings.warn("weight decay only applies for AdamW")

    # scheduler object
    if lr_decay < 1.0:
        lr_scheduler = optax.exponential_decay(lr, 1000, decay_rate=lr_decay, staircase=True)
    else:
        lr_scheduler = optax.constant_schedule(lr)

    # optimizer object
    if optimizer == "SGD":
        opt = optax.sgd(lr_scheduler)
    elif optimizer == "Adam":
        opt = optax.adam(lr_scheduler)
    elif optimizer == "AdamW":
        if mask_fn is None:
            mask_fn = functools.partial(
                hk.data_structures.map,
                lambda _, name, __: name != "__positive_log_scale_param",
            )
        opt = optax.adamw(lr_scheduler, weight_decay=weight_decay, mask=mask_fn)

    optimizer_state = opt.init(parameters)
    return opt, optimizer_state
