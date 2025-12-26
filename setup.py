# Copyright (c) Meta Platforms, Inc. and affiliates.

from distutils.core import setup

from setuptools import find_packages

setup(
    name="aina",
    packages=find_packages(),  # find_packages are not installing any extra packages for now
)
