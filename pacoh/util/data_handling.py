import functools
import warnings

import numpy as np
from typing import Optional
import jax
from jax import numpy as jnp
from numpyro.distributions import Independent, MultivariateNormal, Normal
from torch.utils import data as torch_data
from torch.utils.data.sampler import Sampler

from pacoh.modules.distributions import get_diagonal_gaussian


class DataNormalizer:
    def __init__(self, input_dim, output_dim, normalize_data=True, flatten_ys=False):
        self.x_mean = jnp.zeros((input_dim,))
        self.x_std = jnp.ones((input_dim,))

        self.y_mean = jnp.zeros((output_dim,))
        self.y_std = jnp.ones((output_dim,))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.turn_off_normalization = not normalize_data
        self.flatten_ys = flatten_ys

    @classmethod
    def from_meta_tuples(cls, meta_train_tuples, normalize_data=True, flatten_ys=False):
        """ Initializes a Normalizer based on the provided meta-dataset. """
        xs_stack, ys_stack = map(list,
                                 zip(*[handle_batch_input_dimensionality(x_train, y_train, flatten_ys=flatten_ys)
                                       for x_train, y_train in meta_train_tuples]))
        all_xs, all_ys = np.concatenate(xs_stack, axis=0), np.concatenate(ys_stack, axis=0)
        return cls.from_dataset(all_xs, all_ys, normalize_data, flatten_ys)

    @classmethod
    def from_dataset(cls, xs, ys, normalize_data=True, flatten_ys=False):
        """ Initializes a Normalizer based on the provided dataset. """
        xs, ys = handle_batch_input_dimensionality(xs, ys, flatten_ys=flatten_ys)
        assert not flatten_ys or (flatten_ys and xs.ndim == 2 and ys.ndim == 1), "Something seems off with your data"
        assert flatten_ys or (not flatten_ys and xs.ndim == 2 and ys.ndim == 2), "Something seems off with your data"

        input_dim = xs.shape[-1]
        output_dim = ys.shape[-1]

        normalizer = cls(input_dim, output_dim, normalize_data)
        normalizer.x_mean, normalizer.y_mean = jnp.mean(xs, axis=0), jnp.mean(ys, axis=0)
        normalizer.x_std, normalizer.y_std = jnp.std(xs, axis=0) + 1e-8, jnp.std(ys, axis=0) + 1e-8
        normalizer.flatten_ys = flatten_ys
        return normalizer

    def normalize_data(self, xs, ys=None):
        """
        Normalizes the data according to the stored statistics and returns the normalized data.
        Assumes the data already has the correct dimensionality.
        """
        if self.turn_off_normalization:
            if ys is None:
                return xs
            else:
                return xs, ys

        xs_normalized = (xs - self.x_mean[None, :]) / self.x_std[None, :]  # check whether this is correct
        if ys is None:
            return xs_normalized
        else:
            if not self.flatten_ys:
                ys_normalized = (ys - self.y_mean[0]) / self.y_std[0]
            else:
                ys_normalized = (ys - self.y_mean) / self.y_std
            return xs_normalized, ys_normalized

    def handle_data(self, xs, ys=None):
        if ys is not None:
            xs, ys = handle_batch_input_dimensionality(xs, ys, flatten_ys=self.flatten_ys)
            return self.normalize_data(xs, ys)
        else:
            xs = handle_batch_input_dimensionality(xs, flatten_ys=self.flatten_ys)
            return self.normalize_data(xs)

    def handle_meta_tuples(self, meta_tuples):
        """ Method for convenience. Handles the dimensionality of the meta_tuples and normalizes them based on the
        stored statistics. """
        return [self.handle_data(xs, ys) for xs, ys in meta_tuples]


def normalize_predict(predict_fn):
    """
    Decorator taking the predict method of a RegressionModule as input and outputs the same method
    but applying normalization beforehand and de-normalization afterwards

    Note: when applying this decorator to a method, the resulting method is extended with the argument
        return_density, defaulting to the value True
    """
    def normalized_predict(self, test_x, *, return_density=True, return_full_covariance=True):
        test_x_normalized = self._normalizer.handle_data(test_x)
        pred_dist = predict_fn(self, test_x_normalized)

        if not self._normalizer.turn_off_normalization:
            y_mean = self._normalizer.y_mean
            y_std = self._normalizer.y_std
        else:
            y_mean = jnp.zeros_like(self._normalizer.y_mean)
            y_std = jnp.ones_like(self._normalizer.y_std)

        if return_full_covariance and not return_density:
            warnings.warn("You want the full covariance but only asked for mean and std of individual " +
                          "points... return_density ignored")
            warnings.warn("There is probably a smarter thing to do here")
            return_full_covariance = False

        new_loc = y_std * pred_dist.loc + y_mean

        # Different normalization procedure depending on the type of the underlying predictive distribution
        if isinstance(pred_dist, Independent):
            new_scale = jnp.sqrt(pred_dist.variance) * y_std
            transformed = Normal(new_loc, new_scale)

            assert transformed.variance.shape == pred_dist.variance.shape
            assert transformed.mean.shape == pred_dist.mean.shape

            if return_full_covariance:
                raise ValueError(
                    "Cannot return full covariance if the base learner does not return full covariance. Please debug")

            if return_density:
                return transformed
            else:
                return new_loc, new_scale

        elif isinstance(pred_dist, MultivariateNormal):
            assert self.flatten_ys, "Multivariate normals with multidimensional outputs are not supported (yet?)"
            new_cov = pred_dist.covariance_matrix * y_std ** 2
            transformed = MultivariateNormal(loc=new_loc, covariance_matrix=new_cov)

            if return_full_covariance:
                return transformed

            diag_std = jnp.sqrt(jnp.diag(new_cov))
            pass
            if return_density:
                return get_diagonal_gaussian(new_loc, diag_std)
            else:
                return new_loc, diag_std
        else:
            raise NotImplementedError("This posterior distribution is not supported: " + str(pred_dist.__class__))

    return normalized_predict


def _meta_collate_fn(batch, task_bs, ds_bs):
    xs, ys = jnp.array([x for x, y in batch]), jnp.array([y for x, y in batch])
    return jnp.reshape(xs, (task_bs, ds_bs, -1)), jnp.reshape(ys, (task_bs, ds_bs, -1))


def _flatten_index(task, data_pt, max_len):
    return task*max_len + data_pt


def _unflatten_index(index, max_len):
    return index // max_len, index % max_len


class MetaDataset(torch_data.Dataset):
    """
    A dataset that will allow batching over both the task and dataset level.
        torch.data.Datasets are indexable. In our case, this means transforming a
    """
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
                # get data-points for one of the chosen task
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

        def unzip_collate(batch):
            # batch is a list of tuples of arrays
            # want: a tuple of lists of arrays
            xs_list = [xs for xs, _ in batch]
            ys_list = [ys for _, ys in batch]
            return xs_list, ys_list

        super().__init__(meta_tuples, batch_size=task_batch_size, sampler=sampler, collate_fn=unzip_collate)


def handle_batch_input_dimensionality(xs: np.ndarray, ys: Optional[np.ndarray] = None, flatten_ys: bool = False):
    """
    Takes a dataset S=(xs,ys) and returns it in desired shape whenever possible. Returned values are as follows:
    xs shall have shape (num_points, input_dim) and
    ys shall have size (num_points,) or (num_points, 1)
    Args:
        xs: The inputs
        ys: The labels (optional)
        flatten_ys: Whether to return ys as (num_points), or (num_points, 1)
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
