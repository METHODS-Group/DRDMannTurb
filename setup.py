# read contents of README for PyPi description
from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="drdmannturb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.0.0",
    description="Mann turbulence modelling",
    url="https://github.com/METHODS-Group/DRDMannTurb",
    author="Alexey Izmailov, Matthew Meeker based on code by Brendan Keith et al.",
    author_email="alizma@brown.edu, matthew_meeker@brown.edu",
    license="BSD 2-clause",
    packages=["drdmannturb"],
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "scipy",
        "matplotlib",
        "pyevtk",
        "pyfftw",
        "tensorboard",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
