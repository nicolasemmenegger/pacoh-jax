import jax.nn
import torch
import gpytorch
import math
from collections import OrderedDict
from config import device
from meta_bo.models.util import find_root_by_bounding

from jax.scipy.linalg import cho_factor, cho_solve
from jax import vmap
from jax import numpy as jnp
import haiku as hk
from typing import Tuple, Callable
from abc import ABC, abstractmethod


""" ----------------------------------------------------"""
""" ------------------ Neural Network ------------------"""
""" ----------------------------------------------------"""


class JAXNeuralNetwork(hk.Module):
    def __init__(self,
                 input_dim: int = 2,
                 output_dim: int = 2,
                 layer_sizes: Tuple = (64, 64),
                 nonlinearity: Callable[[float], float] = jax.nn.tanh,
                 weight_norm: bool = False,
                 name: str = None
                 ):
        super().__init__(name=name)
        self.nonlinearity = nonlinearity
        self.n_layers = len(layer_sizes)

    def __call__(self, x):
        pass


class NeuralNetwork(torch.nn.Sequential):
    """Trainable neural network kernel function for GPs."""

    def __init__(self, input_dim=2, output_dim=2, layer_sizes=(64, 64), nonlinearlity=torch.tanh,
                 weight_norm=False, prefix='', ):
        super(NeuralNetwork, self).__init__()
        self.nonlinearlity = nonlinearlity
        self.n_layers = len(layer_sizes)
        self.prefix = prefix  # with the naming of haiku modules I don't think the this is necessary anymore

        if weight_norm:
            _normalize = torch.nn.utils.weight_norm
        else:
            _normalize = lambda x: x

        self.layers = []
        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, self.prefix + 'fc_%i' % (i + 1), _normalize(torch.nn.Linear(prev_size, size)))
            prev_size = size
        setattr(self, self.prefix + 'out', _normalize(torch.nn.Linear(prev_size, output_dim)))

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers + 1):
            output = getattr(self, self.prefix + 'fc_%i' % i)(output)
            output = self.nonlinearlity(output)
        output = getattr(self, self.prefix + 'out')(output)
        return output

    def forward_parametrized(self, x, params):
        output = x
        param_idx = 0
        for i in range(1, self.n_layers + 1):
            output = F.linear(output, params[param_idx], params[param_idx + 1])
            output = self.nonlinearlity(output)
            param_idx += 2
        output = F.linear(output, params[param_idx], params[param_idx + 1])
        return output


""" ----------------------------------------------------"""
""" ------------ Vectorized Neural Network -------------"""
""" ----------------------------------------------------"""

import torch.nn as nn
import torch.nn.functional as F


class VectorizedModel:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def parameter_shapes(self):
        raise NotImplementedError

    def named_parameters(self):
        raise NotImplementedError

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        if len(name.split('.')) == 1:
            setattr(self, name, value)
        else:
            remaining_name = ".".join(name.split('.')[1:])
            getattr(self, name.split('.')[0]).set_parameter(remaining_name, value)

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def set_parameters_as_vector(self, value):
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            if value.ndim == 1:
                self.set_parameter(name, value[idx:idx_next])
            elif value.ndim == 2:
                self.set_parameter(name, value[:, idx:idx_next])
            else:
                raise AssertionError
            idx = idx_next
        assert idx_next == value.shape[-1]


class LinearVectorized(VectorizedModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        self.weight = torch.normal(0, 1, size=(input_dim * output_dim,), device=device, requires_grad=True)
        self.bias = torch.zeros(output_dim, device=device, requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = _kaiming_uniform_batched(self.weight, fan=self.input_dim, a=math.sqrt(5), nonlinearity='tanh')
        if self.bias is not None:
            fan_in = self.output_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.weight.ndim == 2 or self.weight.ndim == 3:
            model_batch_size = self.weight.shape[0]
            # batched computation
            if self.weight.ndim == 3:
                assert self.weight.shape[-2] == 1 and self.bias.shape[-2] == 1

            W = self.weight.view(model_batch_size, self.output_dim, self.input_dim)
            b = self.bias.view(model_batch_size, self.output_dim)

            if x.ndim == 2:
                # introduce new dimension 0
                x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
                # tile dimension 0 to model_batch size
                x = x.repeat(model_batch_size, 1, 1)
            else:
                assert x.ndim == 3 and x.shape[0] == model_batch_size
            # out dimensions correspond to [nn_batch_size, data_batch_size, out_features)
            return torch.bmm(x, W.permute(0, 2, 1)) + b[:, None, :]
        elif self.weight.ndim == 1:
            return F.linear(x, self.weight.view(self.output_dim, self.input_dim), self.bias)
        else:
            raise NotImplementedError

    def parameter_shapes(self):
        return OrderedDict(bias=self.bias.shape, weight=self.weight.shape)

    def named_parameters(self):
        return OrderedDict(bias=self.bias, weight=self.weight)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NeuralNetworkVectorized(VectorizedModel):
    """Trainable neural network that batches multiple sets of parameters. That is, each
    """

    def __init__(self, input_dim, output_dim, layer_sizes=(64, 64), nonlinearlity=torch.tanh):
        super().__init__(input_dim, output_dim)

        self.nonlinearlity = nonlinearlity
        self.n_layers = len(layer_sizes)

        prev_size = input_dim
        for i, size in enumerate(layer_sizes):
            setattr(self, 'fc_%i' % (i + 1), LinearVectorized(prev_size, size))
            prev_size = size
        setattr(self, 'out', LinearVectorized(prev_size, output_dim))

    def forward(self, x):
        output = x
        for i in range(1, self.n_layers + 1):
            output = getattr(self, 'fc_%i' % i)(output)
            output = self.nonlinearlity(output)
        output = getattr(self, 'out')(output)
        return output

    def parameter_shapes(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).parameter_shapes().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).parameter_shapes().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()

        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).named_parameters().items():
                param_dict[layer_name + '.' + name] = param

        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).named_parameters().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


""" Initialization Helpers """


def _kaiming_uniform_batched(tensor, fan, a=0.0, nonlinearity='tanh'):
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)