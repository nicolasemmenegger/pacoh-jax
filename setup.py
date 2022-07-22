import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pacoh_jax",
    version="0.0.1",
    author="Jonas Rothfuss, Nicolas Emmenegger",
    author_email="jonas.rothfuss@gmail.com, nicolaem@ethz.ch",
    description="JAX implementation of PACOH and F-PACOH",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    test_requires=[
        'pytest>=7.1.2',
        'black>=22.3.0'
    ],
    install_requires=[
        'jax>=0.3.0',  # older version of jax doesn't work with numpyro
        'jaxlib>=0.3.0',  # older version of jax doesn't work with numpyro
        'torch>=1.9.0',  # data loaders, pip complains with lower versions
        'dm-haiku>=0.0.6',  # newest version required because of MultiTransform among other things
        'numpy>=1.20.0',  # jax 0.3.0 onwards needs this
        'numpyro>=0.8.0',  # lower version doesn't work
        'optax>=0.1.0',  # optimization, lower versions don't seem to have the exact same interface
        'matplotlib>=2.0.0',  # plotting
    ],
)