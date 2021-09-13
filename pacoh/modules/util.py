from typing import Optional

import jax
import numpy as np
import os
import logging
from absl import flags
import warnings
import torch


def find_root_by_bounding(fun, left, right, eps=1e-6, max_iter=1e4):
    """
    Root finding method that uses selective shrinking of a target interval bounded by left and right
    --> other than the newton method, this method only works for for vectorized univariate functions
    Args:
        fun (callable): function f for which f(x) = 0 shall be solved
        left: (torch.Tensor): initial left bound
        right (torch.Tensor): initial right bound
        eps (float): tolerance
        max_iter (int): maximum iterations
    """

    assert callable(fun)

    n_iter = 0
    approx_error = 1e12
    while approx_error > eps:
        middle = (right + left)/2
        f = fun(middle)

        left_of_zero = (f < 0).flatten()
        left[left_of_zero] = middle[left_of_zero]
        right[~left_of_zero] = middle[~left_of_zero]

        assert torch.all(left <= right).item()

        approx_error = torch.max(torch.abs(right-left))/2
        n_iter += 1

        if n_iter > max_iter:
            warnings.warn("Max_iter has been reached - stopping newton method for determining quantiles")
            return torch.Tensor([np.nan for _ in range(len(left))] )

    return middle

def _handle_point_input_dimensionality(self, x, y):
    # TODO merge with the _util function and just use that

    if x.ndim == 1:
        assert x.shape[-1] == self.input_dim
        x = X.reshape((-1, self.input_dim))

    if isinstance(y, float) or y.ndim == 0:
        y = np.array(y)
        y = y.reshape((1,))
    elif y.ndim == 1:
        pass
    else:
        raise AssertionError('y must not have more than 1 dim')
    return X, y

def _handle_batch_input_dimensionality(xs: np.ndarray, ys: Optional[np.ndarray] = None, flatten_ys: bool = True):
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


        if flatten:
            return xs, ys.flatten()
        else:
            return xs, ys
    else:
        return xs


def get_logger(log_dir=None, log_file='output.log', expname=''):

    if log_dir is None and flags.FLAGS.is_parsed() and hasattr(flags.FLAGS, 'log_dir'):
        log_dir = flags.FLAGS.log_dir

    logger = logging.getLogger('gp-priors')
    logger.setLevel(logging.INFO)

    if len(logger.handlers) == 0:

        #formatting
        if len(expname) > 0:
            expname = ' %s - '%expname
        formatter = logging.Formatter('[%(asctime)s -' + '%s'%expname +  '%(levelname)s]  %(message)s')

        # Stream Handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        logger.propagate = False

        # File Handler
        if log_dir is not None and len(log_dir) > 0:
            fh = logging.FileHandler(os.path.join(log_dir, log_file))
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.log_dir = log_dir
        else:
            logger.log_dir = None
    return logger

# https://github.com/bryanhpchiang/rt/blob/master/utils/transforms.py#L5
def pytrees_stack(pytrees, axis=0):
    results = jax.tree_multimap(
        lambda *values: jax.numpy.stack(values, axis=axis), *pytrees)
    return results


""" ------ Lightweight mltiprocessing utilities ------ """
from multiprocessing import Process
import multiprocessing
import numpy as np

class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          print('task %i of %i'%(n_tasks-len(tasks), n_tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

class LoopExecutor:

    def run(self, target, *args_iter, verbose=False):
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)


        for i, task in enumerate(tasks):
            target(*task)
            if verbose:
                print('task %i of %i'%(n_tasks-len(tasks), n_tasks))

def _start_process(target, args=None):
    if args:
        p = Process(target=target, args=args)
    else:
        p = Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass
