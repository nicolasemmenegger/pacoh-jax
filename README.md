# Meta-BO based on PACOH / F-PACOH
The repository implements the UCB Bayesian optimization algorithm with support for the following models
* Vanilla GP
* PACOH-GP-MAP [(https://arxiv.org/abs/2002.05551)](https://arxiv.org/abs/2002.05551)
* PACOH-GP-SVGD [(https://arxiv.org/abs/2002.05551)](https://arxiv.org/abs/2002.05551)
* F-PACOH-GP [(https://arxiv.org/abs/2106.03195)](https://arxiv.org/abs/2106.03195)

## Installation
To install the minimal dependencies needed to use the meta-learning algorithms, run in the main directory of this repository
```bash
pip install -e .
``` 

Note that if you want jax with gpu support and don't have it installed already, you can additionally run the command 
```bash
pip install --upgrade jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` 
See also the jax documentation for that.

## Usage
First, add the root of this repository to PYTHONPATH, e.g. using `sys.path.append(<path_to_repo>)` within your virtual environment.
Then, simply use `from pacoh.models import ...` to import the model you want to meta-learn with.



Check out the `models` directory for more documentation of how to use the models and the parameters available.

## Basic Usage
First, add the root of this repository to PYTHONPATH. 
The following code snippet provides a basic usage example of F-PACOH:

```python
from pacoh.bo.meta_environment import RandomMixtureMetaEnv
from pacoh.models import F_PACOH_MAP_GP
import numpy as np

# generate some meta train and meta test data
meta_env = RandomMixtureMetaEnv()
num_train_tasks = 20
meta_train_data = meta_env.generate_uniform_meta_train_data(
    num_tasks=num_train_tasks, num_points_per_task=10
)
x_train, y_train, _, __ = meta_env.generate_uniform_meta_valid_data(
    num_tasks=1, num_points_context=10, num_points_test=10
)[0]
x_pred = np.linspace(meta_env.domain.l, meta_env.domain.u, num=100)


# initialize F-PACOH model
gp_model = F_PACOH_MAP_GP(input_dim=1, output_dim=1, domain=meta_env.domain, num_tasks=num_train_tasks, 
                          weight_decay=1e-4, prior_factor=1e-2, task_batch_size=2, lr=1e-3
)

# meta train the model
gp_model.meta_fit(
    meta_train_data, num_iter_fit=200
)

# meta predict
posterior_mean, posterior_std = gp_model.meta_predict(x_train, y_train, x_pred, return_density=False)
```

## Citing
For usage of the algorithms provided in this repo for your research,
we kindly ask you to acknowledge the two papers that formally introduce them:
```bibtex
@InProceedings{rothfuss21pacoh,
  title = 	 {PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees},
  author =       {Rothfuss, Jonas and Fortuin, Vincent and Josifoski, Martin and Krause, Andreas},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9116--9126},
  year = 	 {2021}
 }

@InProceedings{rothfuss2021fpacoh,
  title={Meta-learning reliable priors in the function space},
  author={Rothfuss, Jonas and Heyn, Dominique and Chen, Jinfan and Krause, Andreas},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```