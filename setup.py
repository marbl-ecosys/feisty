#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().strip().split('\n')

with open('README.rst') as f:
    long_description = f.read()

setup(
    maintainer='Matthew Long',
    maintainer_email='mclong@ucar.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
    description='Python implementation for the Fisheries Size and Functional Type model (FEISTY)',
    install_requires=requirements,
    license='MIT license',
    long_description=long_description,
    include_package_data=True,
    keywords='feisty',
    name='feisty',
    packages=find_packages(include=['feisty', 'feisty.*']),
    url='https://github.com/marbl-ecosys/feisty',
    project_urls={
        'Documentation': 'https://github.com/marbl-ecosys/feisty',
        'Source': 'https://github.com/marbl-ecosys/feisty',
        'Tracker': 'https://github.com/marbl-ecosys/feisty/issues',
    },
    zip_safe=False,
)
