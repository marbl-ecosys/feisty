#!/usr/bin/env python

"""
Script that uses dask to run FEISTY in parallel
"""

import os
import time

import dask_mpi
import xarray as xr
import yaml
from dask.distributed import Client

import feisty

#################
# Configure Run #
#################

# First thing to do: initialize dask-mpi
dask_mpi.initialize()

import argparse

parser = argparse.ArgumentParser(description='Offline driver for FEISTY')
parser.add_argument(
    '-f',
    '--yaml_file',
    action='store',
    default='FOSI.yaml',
    dest='yaml_file',
    help='YAML file containing configuration settings for FEISTY',
)
args = parser.parse_args()
if not os.path.isfile(args.yaml_file):
    raise FileNotFoundError(f'"{args.yaml_file}" can not be found')

# read parameters from yaml
with open(args.yaml_file, 'r') as f:
    parameters = yaml.safe_load(f)

ds = feisty.utils.generate_single_ds_for_feisty(
    num_chunks=parameters['num_chunks'],
    forcing_file=parameters['forcing_file'],
    ic_file=parameters['ic_file'],
    forcing_rename=parameters['forcing_rename'],
)

print(ds)
print('\n----\n')

# Generate a template for the output of map_blocks
template = feisty.utils.generate_template(
    ds=ds,
    nsteps=parameters['nyears'] * 365,
    start_date=parameters['start_date'],
    diagnostic_names=parameters['diagnostic_names'],
)

print(template)
print('\n----\n')

###############
# Set up dask #
###############

nsteps = 365 * parameters['nyears']
print(f'Starting compute at {time.strftime("%H:%M:%S")}')
with Client() as c:
    # map_blocks lets us run in parallel over our dask cluster
    ds_out = xr.map_blocks(
        feisty.config_and_run_from_dataset,
        ds,
        args=(
            nsteps,
            parameters['start_date'],
            parameters['ignore_year_in_forcing'],
            parameters['settings_in'],
            parameters['diagnostic_names'],
            parameters['max_output_time_dim'],
            parameters['method'],
        ),
        template=template,
    ).compute()
    # if output_file is .zarr, skip compute() and call to_zarr() here
    # could also write to zarr, then read zarr and write to netcdf

# omit .compute and write to zarr?
if parameters['output_file']:
    if os.path.isfile(parameters['output_file']):
        print(f'{parameters["output_file"]} exists, removing now...')
        os.remove(parameters['output_file'])

    if parameters.get('output_in_2D', False):
        map_ds = feisty.utils.generate_1D_to_2D_pop_map(parameters['forcing_file'])
        ds_out2 = feisty.utils.map_ds_back_to_2D_pop(ds_out, map_ds)
        ds_out2.to_netcdf(parameters['output_file'])
    else:
        ds_out.to_netcdf(parameters['output_file'])

print(f'Finished at {time.strftime("%H:%M:%S")}')

# dask_mpi.send_close_signal()
