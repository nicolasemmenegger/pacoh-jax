# Meta-BO based on PACOH / F-PACOH
The repository implements the UCB Bayesian optimization algorithm with support for the following models
* Vanilla GP
* PACOH-GP-MAP [(https://arxiv.org/abs/2002.05551)](https://arxiv.org/abs/2002.05551)
* PACOH-GP-SVGD [(https://arxiv.org/abs/2002.05551)](https://arxiv.org/abs/2002.05551)
* F-PACOH-GP [(https://arxiv.org/abs/2106.03195)](https://arxiv.org/abs/2106.03195)

## Installation
To install the minimal dependencies needed to use the meta-learning algorithms, run in the main directory of this repository
```bash
pip install .
``` 

For full support of all scripts in the repository, for instance to run the unit tests, further dependencies need to be installed. 
To do so, please run in the main directory of this repository 
```bash
pip install -r requirements.txt
```

## Usage
First, add the root of this repository to PYTHONPATH, e.g. using `sys.path.append(<path_to_repo>)` within your virtual environment.
Then, simply use `from pacoh.models import ...` to import the model you want to meta-learn with.


Check out the `models` directory for how to use the models, by running e.g.
`python pacoh/models/f_pacoh_map.py`

## Contributing: a Note on code formatting in this repository
We use the popular black code formatter. Run 
```bash
black pacoh --line-length 110
``` 
to automatically format all the files in the pacoh directory.