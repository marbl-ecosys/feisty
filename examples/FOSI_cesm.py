#!/usr/bin/env python

"""
Script that uses dask to run FEISTY in parallel
"""

import os
import time

import dask_mpi
import xarray as xr
from dask.distributed import Client

import feisty

#################
# Configure Run #
#################

# First thing to do: initialize dask-mpi
dask_mpi.initialize()

parameters = dict()
parameters[
    'num_chunks'
] = 18  # number of dask chunks to break data into; this is the number of parallel tasks that will be run
parameters[
    'ignore_year_in_forcing'
] = False  # For a spin-up run, generate a dataset with a single year of forcing and set this to true
parameters['nyears'] = 62  # 50 # Length of run (for FOSI, there are 68 years of forcing)
parameters[
    'start_date'
] = '0249-01-01'  # Match calendar for forcing, which kept the CESM mid-month dates from the run: 0249-01-15 through 0316-12-15
parameters[
    'settings_in'
] = {}  # default settings in feisty/core/default_settings.yaml are correct for FOSI
parameters['diagnostic_names'] = []  # only want biomass in output
parameters['max_output_time_dim'] = 365  # break up output into 1-year chunks
parameters['method'] = 'euler'  # only available time-stepping method at this time

# FEISTY has a script that can read forcing / initial condition files and generate the necessary dataset
# (1) provide paths to netcdf files containing forcing and ic
feisty_data_root = os.path.join(os.sep, 'glade', 'work', 'mlevy', 'codes', 'feisty', 'input_files')
parameters['forcing_file'] = os.path.join(feisty_data_root, 'feisty_input_from_FOSI_monthly.nc')
parameters['ic_file'] = os.path.join(feisty_data_root, 'FOSI_cesm_init_200yr.nc')
parameters['output_file'] = f'FOSI_cesm.{parameters["nyears"]}_yr.nc'
if os.path.isfile(parameters['output_file']):
    print(f'{parameters["output_file"]} exists, removing now...')
    os.remove(parameters['output_file'])

# (2) provide a dictionary containing any variables that need to be renamed
forcing_rename = dict()
forcing_rename['time'] = 'forcing_time'
forcing_rename['dep'] = 'bathymetry'
ds = feisty.utils.generate_single_ds_for_feisty(
    num_chunks=parameters['num_chunks'],
    forcing_file=parameters['forcing_file'],
    ic_file=parameters['ic_file'],
    forcing_rename=forcing_rename,
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

# sys.exit(0)
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

ds_out.to_netcdf(parameters['output_file'])

print(f'Finished at {time.strftime("%H:%M:%S")}')
print(ds_out.isel(X=55000))

# dask_mpi.send_close_signal()
