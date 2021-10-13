import jax
import jax.numpy as jnp

from pacoh.util.tree import pytree_unstack


def initialize_model(init, *shapes, random=None):
    dummy_data = []
    for shape in shapes:
        dummy_data.append(jnp.ones(shape))

    params = init(random, *dummy_data)
    return params


def initialize_batched_model(init, *shapes, n_models, random=None):
    dummy_data = []
    for shape in shapes:
        dummy_data.append(jnp.ones([n_models, shape]))

    if random is not None:
        random = jax.random.split(random, n_models)

    params = init(random, *dummy_data)
    param_template = pytree_unstack(params)[0]
    return params, param_template


# maybe add an option so that it does apply_rng?