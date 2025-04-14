# read contents of README for PyPi description
import os
from pathlib import Path

from setuptools import setup


def get_package_files(docs_root):
    data_files = []
    for dirname, dirs, files in os.walk(docs_root):
        for f in files:
            data_files.append(os.path.join(dirname, f))
    return data_files


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="drdmannturb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.0.2",
    description="Mann turbulence modelling",
    url="https://github.com/METHODS-Group/DRDMannTurb",
    author="Alexey Izmailov, Matthew Meeker, Yorgos Deskos based on code by Brendan Keith et al.",
    author_email="alizma@brown.edu, meeker.matthew@pm.me",
    license="BSD 2-clause",
    packages=["drdmannturb"],
    include_package_data=True,
    package_data={
        "docs": get_package_files("./docs"),
        "examples": get_package_files("./examples"),
    },
    install_requires=[
        "torch",
        "numpy<2.0",
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
