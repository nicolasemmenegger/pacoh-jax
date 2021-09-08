import functools

import jax
from jax import vmap, numpy as jnp
import haiku as hk

from pacoh.models.pacoh_map_gp import BaseLearnerInterface
from pacoh.modules.distributions import JAXGaussianLikelihood
from pacoh.modules.gp.gp_lib import JAXExactGP
from pacoh.modules.gp.kernels import JAXRBFKernel


def get_batched_module(init_fn, apply_fns, multi=False, with_state=False):
    """ Takes an init function and either a single apply funciton or a tuple thereof, and returns
        batched module versions of them. This means it initialises a number of models in paralels
        Args:
            init:  a haiku init function (key, data) -> params
            apply: a tuple or apply functions (params, key, data) -> Any
        Returns:
            init_batcbed: ([keys], data) -> [params]
            apply_batched: ([params], [keys], data) -> [Any]
            apply_batched_batched_inputs:  ([params], [keys], [data]) -> [Any]
        Notes:
            apply_fns is expected to be a named tuple, in which case this method returns named tuples of the same type
            init batched takes a number of PRNGKeys and one example input and returns that number of models
            apply_batched takes a number of parameter trees and same number of keys and one data batch and returns that number of inputs
            apply_batched_batched_inputs will assume that the data is batched on a per_model baseis, i.e. that every model gets
                a different data batch. This corresponds to batch_inputs=True in the call method in Jonas' code
       """

    """batched.init returns a batch of model parameters, which is why it takes n different random keys"""
    batched_init = vmap(init_fn, in_axes=(0, None))
    # if our transformed modules use hk.get_state and hk.set_state, we have to have one more vmapped dimension
    if with_state:
        batched_dims = (0, 0, 0, None)
        batched_dims_batch_inputs = (0, 0, 0, 0)
    else:
        batched_dims = (0, 0, None)
        batched_dims_batch_inputs = (0, 0, 0)

    if not multi:
        # there is only one apply function
        apply_batched = vmap(apply_fns, in_axes=batched_dims)
        apply_batched_batched_inputs = vmap(apply_fns, in_axes=batched_dims_batch_inputs)
        return batched_init, apply_batched, apply_batched_batched_inputs
    else:
        # there are multiple apply functions (see also hk.multi_transform and hk.multi_transform_with_state)
        apply_dict = {}
        apply_dict_batched_inputs = {}
        for fname, func in apply_fns._asdict().items():
            apply_dict[fname] = vmap(func, in_axes=batched_dims)
            apply_dict_batched_inputs[fname]  = vmap(func, in_axes=batched_dims_batch_inputs)

        return batched_init, apply_fns.__class__(**apply_dict), apply_fns.__class__(**apply_dict_batched_inputs)

""" ---- Decorators ----- """
def _transform_batch_base(constructor_fn, multi=False, with_state=False):
    """ Decorates a factory which returns the argument to one of the haiku transforms, depending on whether multi
    and with state are true or false """
    if not multi:
        if not with_state:
            purify_fn = hk.transform
        else:
            purify_fn = hk.transform_with_state
    else:
        if not with_state:
            purify_fn = hk.multi_transform
        else:
            purify_fn = hk.multi_transform_with_state

    def batched_constructor_fn(*args, **kwargs):
        return get_batched_module(*purify_fn(constructor_fn(*args, **kwargs)), multi, with_state)
    return batched_constructor_fn


transform_and_batch_module = _transform_batch_base
transform_and_batch_module_with_state = functools.partial(_transform_batch_base, multi=False, with_state=True)
multi_transform_and_batch_module = functools.partial(_transform_batch_base, multi=True, with_state=False)
multi_transform_and_batch_module_with_state = functools.partial(_transform_batch_base, multi=True, with_state=True)


""" ------ Testing code ------- """
if __name__ == "__main__":
    # some testing code
    @transform_and_batch_module
    def get_pure_batched_nn_functions(output_dim, hidden_layer_sizes, activation):
        def forward(xs):
            nn = hk.nets.MLP(output_sizes=hidden_layer_sizes+(output_dim,), activation=activation)
            return nn(xs)
        return forward

    @multi_transform_and_batch_module_with_state
    def batched_pacoh_gp_map_forward(input_dim):
        def factory():
            covar_module = JAXRBFKernel(input_dim)
            mean_module = hk.nets.MLP(output_sizes=(32,32) + (1,), activation=jax.nn.tanh)
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
    outs = batched_forward(all_params, init_keys, xs1)
    print("1ton", outs)
    xs2 = jnp.array([[[1]], [[2]], [[1]]])
    outs2 = batched_forward_batch_inputs(all_params, init_keys, xs2)
    print("1to1", outs2)


    print("TESTING multi tranform and batch with state ")
    gpinit, gpapplys, gpapplys_batched = batched_pacoh_gp_map_forward(1)
    print(gpinit)
    print(gpapplys)

    params, state = gpinit(init_keys, xs_single)
    print("gp params", params)
    print("gp states", state)
    output, state = gpapplys.base_learner_predict(params, state, init_keys, xs1)
    print("It's very nice, because I get the output of 3 gps here", output.loc)


