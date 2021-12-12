import unittest

from jax import numpy as jnp

from pacoh.util.data_handling import MetaDataLoaderTwoLevel, MetaDataLoaderOneLevel


class DataLoaderTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ar = jnp.arange
        self.meta_tuples = [(ar(101, 109), ar(111, 119)), (ar(201, 209), ar(211, 219)), (ar(301, 307), ar(311, 317)),
                            (ar(401, 408), ar(411, 418)), (ar(501, 507), ar(511, 517))]

        self.task_batch_size = 2
        self.dataset_batch_size = 3
        self.num_batches_i_want = 41

        self.dataloader_one = MetaDataLoaderOneLevel(self.meta_tuples, 2, iterations=self.num_batches_i_want)
        self.dataloader_two = MetaDataLoaderTwoLevel(self.meta_tuples, 2, 3, iterations=self.num_batches_i_want)

    def test_num_iterations(self):
        count = 0
        for _ in self.dataloader_one:
            count += 1
        self.assertEqual(count, self.num_batches_i_want)

        count = 0
        for _ in self.dataloader_two:
            count += 1
        self.assertEqual(count, self.num_batches_i_want)

    def test_batch_size(self):
        for xs_batch, ys_batch in self.dataloader_one:
            self.assertEqual(len(xs_batch), self.task_batch_size)
            self.assertEqual(len(ys_batch), self.task_batch_size)
            shapes = [(xs.shape, ys.shape) for xs, ys in self.meta_tuples]

            for xs, ys in zip(xs_batch, ys_batch):
                self.assertIn((xs.shape, ys.shape), shapes)

        for xs, ys in self.dataloader_two:
            self.assertEqual(xs.shape, (self.task_batch_size, self.dataset_batch_size, 1))
            self.assertEqual(ys.shape, (self.task_batch_size, self.dataset_batch_size, 1))

