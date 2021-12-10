import functools
import itertools
import warnings

import numpy as np
from typing import Optional
import jax
from jax import numpy as jnp
from torch.utils import data as torch_data
from torch.utils.data.sampler import Sampler

from pacoh.modules.distributions import AffineTransformedDistribution
from pacoh.util.typing import RawPredFunc, NormalizedPredFunc


class DataNormalizer:
    def __init__(self, input_dim, output_dim, normalize_data=True):
        self.x_mean = jnp.zeros((input_dim,))
        self.x_std = jnp.ones((input_dim,))

        self.y_mean = jnp.zeros((output_dim,))
        self.y_std = jnp.ones((output_dim,))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.turn_off_normalization = not normalize_data

    @classmethod
    def from_meta_tuples(cls, meta_train_tuples, normalize_data=True):
        xs_stack, ys_stack = map(list,
                                 zip(*[handle_batch_input_dimensionality(x_train, y_train) for x_train, y_train in
                                       meta_train_tuples]))
        all_xs, all_ys = np.concatenate(xs_stack, axis=0), np.concatenate(ys_stack, axis=0)
        return cls.from_dataset(all_xs, all_ys, normalize_data)

    @classmethod
    def from_dataset(cls, xs, ys, normalize_data=True):
        """
        Computes mean and std of the dataset and sets the statistics
        """
        xs, ys = handle_batch_input_dimensionality(xs, ys, flatten_ys=False)
        assert xs.ndim == ys.ndim == 2, "Something seems off with your data"
        input_dim = xs.shape[-1]
        output_dim = ys.shape[-1]

        normalizer = cls(input_dim, output_dim, normalize_data)
        normalizer.x_mean, normalizer.y_mean = jnp.mean(xs, axis=0), jnp.mean(ys, axis=0)
        normalizer.x_std, normalizer.y_std = jnp.std(xs, axis=0) + 1e-8, jnp.std(ys, axis=0) + 1e-8

        return normalizer

    def normalize_data(self, xs, ys=None):
        """
        Normalizes the data according to the stored statistics and returns the normalized data
        Assumes the data already has the correct dimensionality.
        """
        # # TODO rermove
        if ys is None:
            return xs
        else:
            return xs, ys

        if self.turn_off_normalization:
            if ys is None:
                return xs
            else:
                return xs, ys

        xs_normalized = (xs - self.x_mean[None, :]) / self.x_std[None, :]

        if ys is None:
            return xs_normalized
        else:
            ys_normalized = (ys - self.y_mean[None, :]) / self.y_std[None, :]
            return xs_normalized, ys_normalized

    def handle_data(self, xs, ys=None):
        if ys is not None:
            xs, ys = handle_batch_input_dimensionality(xs, ys, flatten_ys=False)
            return self.normalize_data(xs, ys)
        else:
            xs = handle_batch_input_dimensionality(xs)
            return self.normalize_data(xs)

    def handle_meta_tuples(self, meta_tuples):
        return list(map(lambda tup: self.handle_data(tup[0], tup[1]), meta_tuples))


def normalize_predict(predict_fn: RawPredFunc) -> NormalizedPredFunc:
    """
    Important note: when applying this decorator to a method, the resulting method is extended with the argument
        return_density, defaulting to the value True
    """
    return predict_fn
    def f(x, b=True):
        return predict_fn(x)

    return f


    def normalized_predict(self, test_x, return_density=True, *args):
        test_x_normalized = self._normalizer.handle_data(test_x)
        pred_dist = predict_fn(self, test_x_normalized, *args)

        if not self._normalizer.turn_off_normalization:
            mean = self._normalizer.y_mean
            std = self._normalizer.y_std
        else:
            mean = jnp.zeros_like(self._normalizer.y_mean)
            std = jnp.ones_like(self._normalizer.y_std)

        pred_dist_transformed = AffineTransformedDistribution(pred_dist,
                                                              normalization_mean=mean,
                                                              normalization_std=std)

        if return_density:
            return pred_dist_transformed.iid_normal
        else:
            return pred_dist_transformed.mean, pred_dist_transformed.stddev

    return normalized_predict


def _meta_collate_fn(batch, task_bs, ds_bs):
    xs, ys = jnp.array([x for x,y in batch]), jnp.array([y for x, y in batch])
    return jnp.reshape(xs, (task_bs, ds_bs, -1)), jnp.reshape(ys, (task_bs, ds_bs, -1))


def _flatten_index(task, data_pt, max_len):
    return task*max_len + data_pt


def _unflatten_index(index, max_len):
    return index // max_len, index % max_len


class MetaDataset(torch_data.Dataset):
    """ A dataset that will allow baatching over both the task and dataset level"""
    def __init__(self, meta_tuples):
        self.meta_tuples = meta_tuples
        self.len_per_task = [xs.shape[0] for xs, ys in meta_tuples]
        self.max_len = max(self.len_per_task)
        self.num_tasks = len(meta_tuples)

    def __getitem__(self, item):
        task_ind, point_ind = _unflatten_index(item, self.max_len)
        xs, ys = self.meta_tuples[task_ind]
        return xs[point_ind], ys[point_ind]

    def __len__(self):
        return sum(self.len_per_task)


class MetaBatchSamplerWithReplacement(Sampler):
    def __init__(self, dataset, task_batch_size, dataset_batch_size, total_iterations=None, random_state=None):
        """
        :param dataset: A dataset of meta_train_tuples
        :param task_batch_size: The number of tasks in a batch
        :param dataset_batch_size: The batch size **within** a  task
        :param return_list: If False, will return a fully vectorizable jax.numpy.array
        :return:
        """
        super().__init__(dataset)
        if not isinstance(dataset, MetaDataset):
            raise ValueError("Can only instantiate TwoLevelBatchSampler with a MetaDataset")
        self.dataset = dataset
        self.task_batch_size = task_batch_size
        self.dataset_batch_size = dataset_batch_size
        self.rds = jax.random.PRNGKey(42) if random_state is None else random_state
        self.total_iterations = 1e100 if total_iterations is None else total_iterations

    def __iter__(self):
        for _ in range(self.total_iterations):
            self.rds, choice_key, *dataset_keys = jax.random.split(self.rds, 2+self.task_batch_size)
            batch_indices = jax.random.choice(choice_key, self.dataset.num_tasks, shape=(self.task_batch_size,))
            indices_list = []
            for i, task in enumerate(batch_indices):
                # get datapoints for one of the chosen task
                task_indices = jax.random.choice(dataset_keys[i],
                                                 self.dataset.len_per_task[task],
                                                 shape=(self.dataset_batch_size,))

                indices_list.append([_flatten_index(task, point, self.dataset.max_len) for point in task_indices])

            yield [point for sublist in indices_list for point in sublist]

    def __len__(self):
        return self.total_iterations



class MetaDataLoaderTwoLevel(torch_data.DataLoader):
    """ A dataloader that provides batching over both the task and the dataset.
        Notes:
            A batch consists of a tuple of two three dimensional arrays
    """
    def __init__(self, meta_tuples, task_batch_size, dataset_batch_size, iterations):
        """
        :param meta_tuples: The meta tuples given a list of tuples of jax.numpy.ndarrays
        :param task_batch_size: The number of tasks in a batch
        :param dataset_batch_size: The number of points per task. If not -1
        :param iterations: The number of batches to deliver. Should correspond to the num_iter of the train loop
        """
        dataset = MetaDataset(meta_tuples)
        meta_batch_sampler = MetaBatchSamplerWithReplacement(dataset, task_batch_size,
                                                             dataset_batch_size, total_iterations=iterations)

        super().__init__(dataset,
                         batch_sampler=meta_batch_sampler,
                         collate_fn=functools.partial(_meta_collate_fn,
                                                      task_bs=task_batch_size, ds_bs=dataset_batch_size))


class MetaDataLoaderOneLevel(torch_data.DataLoader):
    """ A dataloader that implements one way batching, namely at the task level.
        Notes:
            A batch consists of two lists of full datasets
    """
    def __init__(self, meta_tuples, task_batch_size, iterations):
        """
        :param meta_tuples: The meta tuples given a list of tuples of jax.numpy.ndarrays
        :param task_batch_size: The number of tasks in a batch
        :param iterations: The number of batches to deliver. Should correspond to the num_iter of the train loop
        """
        sampler = torch_data.RandomSampler(meta_tuples, replacement=True, num_samples=iterations*task_batch_size)
        super().__init__(meta_tuples, batch_size=task_batch_size, sampler=sampler, collate_fn=lambda batch: batch)

def handle_point_input_dimensionality(self, x, y):
    if x.ndim == 1:
        assert x.shape[-1] == self._input_dim
        x = x.reshape((-1, self._input_dim))

    if isinstance(y, float) or y.ndim == 0:
        y = np.array(y)
        y = y.reshape((1,))
    elif y.ndim == 1:
        pass
    else:
        raise AssertionError('y must not have more than 1 dim')
    return x, y


def handle_batch_input_dimensionality(xs: np.ndarray, ys: Optional[np.ndarray] = None, flatten_ys: bool = False):
    """
    Takes a dataset S=(xs,ys) and returns it in a uniform fashion. x shall have shape (num_points, input_dim) and
    y shall have size (num_points), that is, we only consider scalar regression targets.
    Args:
        xs: The inputs
        ys: The labels (optional)
        flatten: Whether to return ys as (num_points), or (num_points, 1)
    Notes:
        ys can be None, to easily handle test data.
    """
    if xs.ndim == 1:
        xs = np.expand_dims(xs, -1)

    assert xs.ndim == 2

    if ys is not None:
        if flatten_ys:
            ys = ys.flatten()
            assert xs.shape[0] == ys.size
            return xs, ys
        else:
            if ys.ndim == 1:
                ys = np.expand_dims(ys, -1)
            assert xs.shape[0] == ys.shape[0], "Number of points and labels is not the same"
            assert ys.ndim == 2
            return xs, ys
    else:
        return xs



# class Sampler:
#     def __init__(self, xs, ys, batch_size, rds, shuffle=True):
#         self.num_batches = xs.shape[0] // batch_size
#         self.shuffle = shuffle
#         self._rng = rds
#         if shuffle:
#             self._rng, shuffle_key = jax.random.split(self._rng)
#             ids = jnp.arange(xs.shape[0])
#             perm = jax.random.permutation(shuffle_key, ids)
#             self.i = -1
#             self.xs = xs
#             self.ys = ys
#             warnings.warn("This is definitely wrong")
#         else:
#             self.xs = xs
#             self.ys = ys
#
#         self.batch_size = batch_size
#         warnings.warn(
#             "this currently does not support a scenario in which the dataset size is not divisible by the batch size")
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.shuffle:
#             # just iterate
#             self.i = (self.i + 1) % self.num_batches
#             start = self.i * self.batch_size
#             end = start + self.batch_size
#             return self.xs[start:end], self.ys[start:end]
#         else:
#             # just subsample at random using the _rds
#             raise NotImplementedError("only shuffle mode is supported for now")


# """ ----- Some simple data loaders ----- """
# class MetaSampler:
#     def __init__(self, meta_tuples, task_batch_size, rds, shuffle=True, minibatch_at_dataset_level=False, dataset_batch_size=None):
#         if minibatch_at_dataset_level and dataset_batch_size is None:
#             raise AssertionError("Please specify a dataset_batch_size")
#
#         num_tuples = len(meta_tuples)
#         self.num_task_batches = num_tuples // task_batch_size
#         self.shuffle = shuffle
#         self._rng = rds
#         self.task_batch_size = task_batch_size
#
#         if minibatch_at_dataset_level:
#             # initialize one sampler per dataset
#             self._rng, *sampler_keys = jax.random.split(self._rng, num_tuples + 1)
#             self.samplers = [Sampler(data[0], data[1], dataset_batch_size, key)
#                              for data, key in zip(meta_tuples, sampler_keys)]
#             self.dataset_batch_size = dataset_batch_size
#
#         self.minibatch_at_dataset_level = minibatch_at_dataset_level
#
#         if shuffle:
#             self._rng, shuffle_key = jax.random.split(self._rng)
#             ids = jnp.arange(len(meta_tuples))
#             perm = jax.random.permutation(shuffle_key, ids)
#             self.i = -1
#             self.xs_list = [meta_tuples[i][0] for i in perm]
#             self.ys_list = [meta_tuples[i][1] for i in perm]
#             self.samplers = [self.samplers[i] for i in perm]
#         else:
#             self.meta_tuples = meta_tuples
#
#         warnings.warn(
#             "this currently does not support a scenario in which the dataset size is not divisible by the batch size")
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.shuffle:
#             # just iterate
#             self.i = (self.i + 1) % self.num_task_batches
#             start = self.i * self.task_batch_size
#             end = start + self.task_batch_size
#             if self.minibatch_at_dataset_level:
#                 subsampled_tuples = [next(sampler) for sampler in self.samplers[start:end]]
#                 array_xs = jnp.stack([tup[0] for tup in subsampled_tuples], axis=0)
#                 array_ys = jnp.stack([tup[1] for tup in subsampled_tuples], axis=0)
#
#                 return array_xs, array_ys
#             else:
#                 # just return a list of tasks
#                 return self.xs_list[start:end], self.ys_list[start:end]
#         else:
#             # just subsample at random using the _rds
#             raise NotImplementedError("only shuffle mode is supported for now")
