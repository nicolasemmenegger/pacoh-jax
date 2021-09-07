import tensorflow as tf



class BatchedModule(tf.Module):
    """Base class for batched fully connected NNs."""

    def __init__(self, n_batched_models=None):
        super().__init__()
        self._variable_sizes = None
        self._parameters_shape = None
        self._n_batched_models = n_batched_models

    def __call__(self, x, **kwargs):
        raise NotImplementedError

    # @tf.function
    def get_variables_stacked_per_model(self):
        vectorized_vars = [tf.reshape(v, (1, -1)) for v in self.variables]
        vectorized_vars = tf.concat(vectorized_vars, axis=1)
        return tf.reshape(vectorized_vars, (self._n_batched_models, -1))

    @tf.function
    def _variable_sizes_method(self):
        if self._variable_sizes is None:
            self._variable_sizes = [tf.size(v) for v in self.variables]
        return self._variable_sizes

    @tf.function
    def _set_variables_vectorized(self, parameters):
        if self._parameters_shape is None:
            self._parameters_shape = parameters.shape

        parameters = tf.reshape(parameters, (-1, 1))
        split = tf.split(parameters, self._variable_sizes_method())

        for v, n_v in zip(self.variables, split):
            v.assign(tf.reshape(n_v, v.shape))

    @tf.function
    def concat_and_vectorize_grads(self, gradients):
        # I am not sure what this does
        vectorized_gradients = tf.concat([tf.reshape(g, (-1, 1)) for g in gradients], axis=0)
        if self._parameters_shape is None:
            return tf.reshape(vectorized_gradients, (self._n_batched_models, -1))
        return tf.reshape(vectorized_gradients, self._parameters_shape)

    @tf.custom_gradient
    def call_parametrized(self, x, variables_vectorized):
        # this is value and grad
        self._set_variables_vectorized(variables_vectorized)

        tape = tf.GradientTape(persistent=True)
        with tape:
            tape.watch([x] + list(self.trainable_variables))
            y = self(x)

        def grad_fn(dy, variables):
            with tape:
                tampered_y = y * dy
            grads_x_w = tape.gradient(tampered_y, [x] + list(self.trainable_variables))
            grads_to_input = [grads_x_w[0], self.concat_and_vectorize_grads(grads_x_w[1:])]
            return grads_to_input, [None] * len(variables)

        return y, grad_fn


class MLP(BatchedModule):
    def __init__(self, input_size, output_size, hidden_sizes, activation=None):
        super().__init__()
        self.n_hidden_layers = len(hidden_sizes)
        self.hidden_layers = []
        for hidden_size in hidden_sizes:
            hidden_layer = Dense(input_size, hidden_size, activation)
            self.hidden_layers.append(hidden_layer)
            input_size = hidden_size

        # no activation for the last layer
        self.output_layer = Dense(input_size, output_size, bias=False)

    # @tf.function
    def __call__(self, x, **kwargs):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)


class BatchedMLP(BatchedModule):
    def __init__(self, n_batched_models, input_size, output_size, hidden_sizes, activation=None):
        super().__init__(n_batched_models)
        self.n_batched_models = n_batched_models
        self.models = []
        for i in range(n_batched_models):
            self.models.append(MLP(input_size, output_size, hidden_sizes, activation))

    # @tf.function
    def __call__(self, inputs, batched_input=False):
        # this calls the n_batchedmodels on all inputs
        if batched_input:
            # inputs: (n_batched_models, batch_size, input_shape)
            tf.assert_equal(tf.rank(inputs), 3)
            tf.assert_equal(inputs.shape[0], self.n_batched_models)
            # this calls each of the n_batched_models on each of the inputs
            outputs = tf.stack([self.models[i](inputs[i]) for i in range(self.n_batched_models)])
        else:
            # inputs: (batch_size, input_shape)
            tf.assert_equal(tf.rank(inputs), 2)
            outputs = tf.stack([self.models[i](inputs) for i in range(self.n_batched_models)])
        return outputs  # (n_batched_models, batch_size, output_shape)