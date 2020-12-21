#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'ml_model'

NAME = 'ml_model'
DESCRIPTION = 'Package to manage the end-to-end machine learning deploy process.'
URL = 'https://github.com/creyesp/ml_to_production'
EMAIL = 'cesar.reyesp@gmail.com'
AUTHOR = 'César Reyes'
REQUIRES_PYTHON = '>=3.6.0'


def list_requirements(file_name='requirements.txt'):
    with open(file_name) as fp:
        return fp.read().splitlines()


try:
    with open(os.path.join(ROOT_DIR, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


with open(os.path.join(PACKAGE_DIR, 'VERSION')) as f:
    package_version = f.read().strip()


setup(
    name=NAME,
    version=package_version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'regression_model': ['VERSION']},
    install_requires=list_requirements(),
    extras_require={},
    include_package_data=True,
)
