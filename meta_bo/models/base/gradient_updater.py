import functools
import warnings
from typing import Any, Collection, Mapping
import jax
from jax import numpy as jnp
import optax
import haiku as hk

# This is modified from https://theaisummer.com/jax-transformer/
# Another way to do this would have been by taking it from https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py
# The named tuple there is quite elegant


class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    def __init__(self, model_init, model_apply,
                 optimizer: optax.GradientTransformation):
        """
        Args:
            model_init: The init_fn function returned by the haiku transform
            model_apply: Tha apply_fn function returned by the haiku transform
        """
        self._model_init = model_init
        self._model_apply = model_apply
        self._optimizer = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data_x, data_y):
        """
        A wrapper to call the haiku.transform.init_fn.
        Initializes the parameters in the involved modules based on some dummy data
        """
        out_rng, init_rng = jax.random.split(master_rng)
        params, hk_state = self._model_init(init_rng, data_x)
        warnings.warn("maybe here we should do something about the different LR schedules")
        opt_state = self._optimizer.init(params)
        out = dict(
            rng=out_rng,
            opt_state=opt_state,
            params=params,
            hk_state=hk_state
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data_x, data_y):
        """
        Args:
            state: A dictionary storing the optimizer state, the current random number generator key and most importantly
                the parameters for all involved modules
            data: a list of datasets

        Returns:
            state: The updated state
            metrics: the results from the foward function
        """
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        hk_state = state['hk_state']
        # has aux is true because our functions return state
        output, gradients = jax.value_and_grad(self._model_apply, has_aux=True)(params, hk_state, rng, data_x, data_y) # actually does not need anything I think
        neg_mle, new_hk_state = output # I think
        updates, new_opt_state = self._optimizer.update(gradients, state['opt_state'], params=params)  # compute the update based on the gradients and optimizer state
        new_params = optax.apply_updates(params, updates)  # apply the computed update to the model's parameters

        new_state = {
            'rng': new_rng,
            'opt_state': new_opt_state,
            'params': new_params,
            'hk_state': new_hk_state
        }

        warnings.warn("I need to do the metrics differently")
        metrics = {
            'negative_mle': neg_mle,
        }
        return new_state, metrics

