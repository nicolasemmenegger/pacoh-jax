import functools
from typing import Optional, Dict, Union

import jax
from haiku import Transformed, TransformedWithState, MultiTransformed, MultiTransformedWithState
from jax import vmap, numpy as jnp
import haiku as hk

from pacoh.models.pure.pure_interfaces import BaseLearnerInterface
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.modules.exact_gp import JAXExactGP
from pacoh.modules.kernels import JAXRBFKernel


def get_batched_module(transformed: Union[Transformed, MultiTransformed, TransformedWithState, MultiTransformedWithState],
                       num_data_args: Optional[Union[int, Dict[str, int]]]):
    """ Takes an init function and either a single apply funciton or a tuple thereof, and returns
        batched module versions of them. This means it initialises a number of models in parallel
        Args:
            transformed: on of Transformed, MultiTransformed, TransformedWithState, MultiTransformedWithState returned
                by haiku
            multi: whether we get a MultiTransformed instance
            num_data_args:  a dictionary that specifies how many data arguments each of the apply functions have.
                The transformed.init function is curerntly constrained to be of type (keys, data)
        Returns:
            init_batched: ([keys], data) -> [params]
            apply_batched: ([params], [keys], data) -> [Any]
            apply_batched_batched_inputs:  ([params], [keys], [data]) -> [Any]
        Notes:
            apply_fns is expected to be a named tuple, in which case this method returns named tuples of the same type
            init batched takes a number of PRNGKeys and one example input and returns that number of models
            apply_batched takes a number of parameter trees and same number of keys and one data batch and returns that number of inputs
            apply_batched_batched_inputs will assume that the data is batched on a per_model baseis, i.e. that every model gets
                a different data batch. This corresponds to batch_inputs=True in the call method in Jonas' code
            read more at https://jax.readthedocs.io/en/latest/pytrees.html to understand how vmap gets applied on the
            parameter and state PyTrees
        Important
            1. for all the functions in apply_fns, it is assumed that their signature is
            (params, [state], rng, data). If data is a pytree, e.g. a tuple, vmap semantics is to map over the leaves
            If the function is instead of type (params, [state], rng, data), fine grained control over how to vmap
            is done via the argument apply_fn_axes={'name': arguments tuple}.
            2. We do not currently support calling the init or apply functions with keyword arguments, as jax.vmap is
            not there yet https://github.com/google/jax/issues/7465
        Debatable
            whether init should work on batched data or not (currently not), then its signature would become
            init_batched: ([keys], [data]) -> [params]
       """

    """batched.init returns a batch of model parameters, which is why it takes n different random keys"""
    multi = isinstance(transformed, MultiTransformed) or isinstance(transformed, MultiTransformedWithState)
    with_state = isinstance(transformed, TransformedWithState) or isinstance(transformed, MultiTransformedWithState)
    init_fn, apply_fns = transformed
    if num_data_args is None:
        num_data_args = 1

    default_args = num_data_args if isinstance(num_data_args, int) else 1
    if multi:
        data_in_axes = {key: (None,) * default_args for key in apply_fns._asdict().keys()}
        data_in_axes_batched = {key: (0,) * default_args for key in apply_fns._asdict().keys()}
    else:
        data_in_axes = (None,) * default_args
        data_in_axes_batched = (0,) * default_args

    if multi:
        if isinstance(num_data_args, dict):
            # override special cases
            for key, val in num_data_args.items():
                data_in_axes[key] = (None,)*val
                data_in_axes_batched[key] = (0,)*val

    batched_init = vmap(init_fn, in_axes=(0, None))  # vmap along the rng dimension
    base_in_axes = (0, 0, 0) if with_state else (0, 0)

    if not multi:
        # there is only one apply function
        apply_batched = vmap(apply_fns, in_axes=base_in_axes + data_in_axes)
        apply_batched_batched_inputs = vmap(apply_fns, in_axes=base_in_axes + data_in_axes_batched)
        return batched_init, apply_batched, apply_batched_batched_inputs
    else:
        # there are multiple apply functions (see also hk.multi_transform and hk.multi_transform_with_state)
        apply_dict = {}
        apply_dict_batched_inputs = {}
        for fname, func in apply_fns._asdict().items(): # TODO is there a public interface to this
            apply_dict[fname] = vmap(func, in_axes=base_in_axes + data_in_axes[fname])
            apply_dict_batched_inputs[fname] = vmap(func, in_axes=base_in_axes + data_in_axes_batched[fname])

        return batched_init, apply_fns.__class__(**apply_dict), apply_fns.__class__(**apply_dict_batched_inputs)


def _transform_batch_base(constructor_fn, purify_fn=hk.transform, num_data_args=None):
    """ Decorates a factory which returns the argument to one of the haiku transforms, depending on whether multi
    and with state are true or false """
    def batched_constructor_fn(*args, **kwargs):
        return get_batched_module(purify_fn(constructor_fn(*args, **kwargs)), num_data_args)
    return batched_constructor_fn


transform_and_batch_module = _transform_batch_base
transform_and_batch_module_with_state = functools.partial(_transform_batch_base, purify_fn=hk.transform_with_state)
multi_transform_and_batch_module = functools.partial(_transform_batch_base, purify_fn=hk.multi_transform)
multi_transform_and_batch_module_with_state = functools.partial(_transform_batch_base, purify_fn=hk.multi_transform_with_state)


if __name__ == "__main__":
    # config.update("jax_debug_nans", True)
    # config.update('jax_disable_jit', True)

    # some testing code
    @transform_and_batch_module
    def get_pure_batched_nn_functions(output_dim, hidden_layer_sizes, activation):
        def forward(xs):
            nn = hk.nets.MLP(output_sizes=hidden_layer_sizes+(output_dim,), activation=activation)
            return nn(xs)
        return forward


    @functools.partial(multi_transform_and_batch_module_with_state,
                       num_data_args={'base_learner_fit': 2, 'base_learner_mll_estimator': 2})
    def batched_pacoh_gp_map_forward(input_dim):
        def factory():
            covar_module = JAXRBFKernel(input_dim)
            mean_module = hk.nets.MLP(output_sizes=(32, 32) + (1,), activation=jax.nn.tanh)
            likelihood = JAXGaussianLikelihood(variance_constraint_gt=1e-3)
            base_learner = JAXExactGP(mean_module, covar_module, likelihood)

            return base_learner.init_fn, BaseLearnerInterface(base_learner_fit=base_learner.fit,
                                                              base_learner_predict=base_learner.pred_dist,
                                                              base_learner_mll_estimator=base_learner.marginal_ll)
        return factory


    print("TESTING transform and batch")

    rds = jax.random.PRNGKey(42)
    batch_size_vi = 3
    init_batched, batched_forward, batched_forward_batch_inputs = get_pure_batched_nn_functions(1, (32, 32), jax.nn.elu)
    print(init_batched)
    keys = jax.random.split(rds, batch_size_vi + 1)
    rds = keys[0]
    init_keys = keys[1:]
    xs_single = jnp.ones((1, 1), dtype=jnp.float32)
    all_params = init_batched(init_keys, xs_single)
    xs1 = jnp.array([[1], [2]])
    ys1 = jnp.array([3,4])
    outs = batched_forward(all_params, init_keys, xs1)
    print("1ton", outs)
    xs2 = jnp.array([[[1]], [[2]], [[1]]])
    outs2 = batched_forward_batch_inputs(all_params, init_keys, xs2)
    print("1to1", outs2)

    xslarge = jnp.array([[1], [2], [3], [4]])
    yslarge = jnp.array([3, 4, 5, 6])


    print("TESTING multi tranform and batch with state ")
    gpinit, gpapplys, gpapplys_batched = batched_pacoh_gp_map_forward(1)
    print(gpinit)
    print(gpapplys)

    params, state = gpinit(init_keys, xs_single)
    print("gp params", params)
    print("gp states", state)
    output, state = gpapplys.base_learner_predict(params, state, init_keys, xs1)
    print("It's very nice, because I get the output of 3 gps here: PRIOR\n", output.loc)
    output, state = gpapplys.base_learner_mll_estimator(params, state, init_keys, xs1, ys1)
    print("MLLestimator", output)
    output, state = gpapplys.base_learner_fit(params, state, init_keys, xs1, ys1)
    output, state = gpapplys.base_learner_predict(params, state, init_keys, xs1)
    print("fitting a second time")
    output, state = gpapplys.base_learner_fit(params, state, init_keys, xslarge, yslarge)
    output, state = gpapplys.base_learner_predict(params, state, init_keys, xslarge)

    print("It's very nice, because I get the output of 3 gps here: POSTERIOR\n", output.loc)

