from setuptools import setup

setup(
    name="drdmannturb",
    version="0.1.0",
    description="Mann turbulence modelling",
    url="https://github.com/mjachi/WindGenerator",
    author="Alexey Izmailov, Matthew Meeker based on code by Brendan Keith et al",
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
