#!/usr/bin/env python

"""
Script that uses dask to run FEISTY in parallel
"""

import os
import tempfile
import time
from subprocess import check_call

import click
import dask_mpi
import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client
from jinja2 import Template

import feisty

path_to_here = os.path.dirname(os.path.realpath(__file__))
USER = os.environ['USER']


class driver_settings(object):
    """Class to support curating FEISTY_driver parameters"""

    def __init__(
        self,
        run_config_file,
        run_settings_file=None,
        continue_run=False,
    ):

        self.run_config_file = run_config_file
        self.run_settings_file = run_settings_file
        self.calendar = 'noleap'

        # set run config properties
        self.config_info = self._read_file(run_config_file, apply_template=True)
        self.settings_in = self._read_file(run_settings_file)

        # self.run_name = config_info.pop('run_name')
        # self.computing_account = config_info.pop('computing_account')
        # self.run_dir_root = config_info.pop('run_dir_root', '.')
        # self.diagnostic_names = config_info.pop('diagnostic_names', [])

        # self.resubmit = config_info.pop('resubmit', False)

        # self.forcing_rename = config_info.pop('forcing_rename', {})
        # self.ignore_year_in_forcing = config_info.pop('ignore_year_in_forcing', False)
        # self.max_output_time_dim = config_info.pop('max_output_time_dim', 365)
        # self.method = config_info.pop('method', 'euler')
        # self.num_chunks = config_info.pop('num_chunks', 18)
        # self.output_in_2D = config_info.pop('output_in_2D', True)
        # self.calendar = config_info.pop('calendar', 'noleap')

        if self.calendar != 'noleap':
            raise NotImplementedError(f'unsupported calendar: {self.calendar}')

        # set ic and forcing properties
        # self.forcing_keys = forcing_info['keys']
        # if continue_run:
        #     assert os.path.exists(self.restart_pointer), 'missing restart pointer file'

        #     restart_info = self._read_file(self.restart_pointer)
        #     self.ic_file = restart_info['ic_file']

        #     assert (
        #         restart_info['forcing_key'] in self.forcing_keys
        #     ), f"unknown forcing key: {restart_info['forcing_key']} in restart pointer"
        #     # increment forcing keys
        #     key_ndx = self.forcing_keys.index(self.forcing_key) + 1

        # else:
        #     # self.ic_file = config_info.pop('ic_file', None)
        #     key_ndx = 0

        # self.forcing_key = self.forcing_keys[key_ndx]
        # self.last_submission = key_ndx >= len(self.forcing_keys) - 1

        # forcing_info_key = forcing_info[self.forcing_key]
        # self.start_date = forcing_info_key.pop('start_date')
        # self.nyears = forcing_info_key.pop('nyears')
        # self.list_forcing_files = forcing_info_key.pop('list_forcing_files')

        # assert not config_info, f'unknown keys in input file: {config_info}'
        # assert not forcing_info_key, f'unknown keys in forcing file: {forcing_info_key}'

    def write_restart_pointer(self):
        """write a restart pointer file"""
        restart_info = {
            'ic_file': self.restart_file,
            'forcing_key': self.forcing_key,
        }
        with open(self.restart_pointer, 'w') as fid:
            yaml.dump(restart_info, fid)

    @property
    def restart_pointer(self):
        return f'{self.run_dir}/restart-pointer.yml'

    def _read_file(self, filename, apply_template=False):
        """return dictionary from YAML file"""
        if filename is None:
            return {}

        # read parameters from yaml
        with open(filename, 'r') as f:
            file_dict_in = yaml.safe_load(f)

        if not apply_template:
            return file_dict_in

        # apply environment variable templating
        file_dict = {}
        for key, value in file_dict_in.items():
            # what's a better way to check for environment variable
            # replacement?
            if isinstance(value, str) and 'env[' in value:
                file_dict[key] = Template(value).render(env=os.environ)
            else:
                file_dict[key] = value

        return file_dict

    @property
    def nsteps(self):
        """return number of timesteps"""
        assert self.nyears is not None, 'nyears not set'
        if self.calendar == 'noleap':
            return 365 * self.nyears
        else:
            raise ValueError(f'unknown calendar: {self.calendar}')

    @property
    def end_date(self):
        return xr.cftime_range(self.start_date, periods=self.nsteps, calendar=self.calendar)[
            -1
        ].strftime('%Y-%m-%d')

    @property
    def file_datestr(self):
        return f"{self.start_date.replace('-', '')}-{self.end_date.replace('-', '')}"

    @property
    def run_dir(self):
        return f'{self.run_dir_root}/{self.run_name}'

    @property
    def output_file(self):
        return f'{self.run_dir}/{self.run_name}.{self.file_datestr}.nc'

    @property
    def restart_file(self):
        return f'{self.run_dir}/{self.run_name}.restart.{self.end_date}.nc'

    def _file_in_run_dir(self, file):
        """return a path to a file copied into `run_dir`"""
        return f'{self.run_dir}/{os.path.basename(file)}'

    def setup_run_dir(self):
        """ensure that files needed to run the model are in `run_dir`"""
        os.makedirs(self.run_dir, exist_ok=True)
        ensure_files = [
            self.run_config_file,
            self.run_settings_file,
        ]
        for file in ensure_files:
            if file is None:
                continue
            check_call(['cp', file, self._file_in_run_dir(file)])


def gen_jobscript(run_config_file, run_settings_file=None, continue_run=False):
    """return a jobscript file suitable for submission to queue"""

    run_settings = driver_settings(
        run_config_file,
        run_settings_file,
        continue_run,
    )

    mpi_tasks = run_settings.num_chunks + 2

    run_config_file = run_settings.run_config_file
    run_settings_file = run_settings.run_settings_file

    indent_spc = ' ' * 4
    options = [
        f'{indent_spc}--run-config-file {run_config_file}',
    ]

    if run_settings.run_settings_file is not None:
        options.append(f'--run-settings-file {run_settings_file}')

    if continue_run:
        options.append('--continue-run')

    # options_str = f' \\ \n{indent_spc}'.join(options)
    options_str = ' '.join(options)

    lines = [
        '#!/bin/bash -l',
        f'#PBS -N {run_settings.run_name}',
        f'#PBS -A {run_settings.computing_account}',
        f'#PBS -l select=1:ncpus={mpi_tasks}:mpiprocs={mpi_tasks}:mem=360GB',
        '#PBS -l walltime=06:00:00',
        '#PBS -q casper',
        '#PBS -j oe',
        '#PBS -m abe',
        '',
        'source activate dev-feisty',
        '',
        f'mpirun -n {mpi_tasks} {path_to_here}/./FEISTY_driver.py {options_str}',
        '',
    ]

    _, jobscript = tempfile.mkstemp(prefix='feisty-run.', suffix='.sh', text=True)

    with open(jobscript, 'w') as fid:
        fid.write('\n'.join(lines))

    return jobscript


def submit_run(run_config_file, run_settings_file=None, continue_run=False):
    """submit a FEISTY integration to the queue"""

    run_settings = driver_settings(
        run_config_file,
        run_settings_file,
        continue_run,
    )

    run_settings.setup_run_dir()

    jobscript = gen_jobscript(run_config_file, run_settings_file, continue_run)

    print('submitting run')
    print(f'qsub {jobscript}')

    return check_call(['qsub', jobscript], cwd=run_settings.run_dir)


@click.command(help='Offline driver for FEISTY')
@click.option(
    '--run-config-file',
    type=click.Path(exists=True),
    required=True,
    help='YAML file containing run settings',
)
@click.option(
    '--run-settings-file',
    type=click.Path(exists=True),
    required=False,
    help='YAML file containing FEISTY model parameters',
)
@click.option(
    '--continue-run',
    is_flag=True,
    required=False,
    help='flag to read initial conditions from restart file',
)
def main(
    run_config_file,
    run_settings_file=None,
    continue_run=False,
):

    # configure Run
    run_settings = driver_settings(
        run_config_file,
        run_settings_file,
        continue_run,
    )
    ds_out = feisty.config_and_run_from_yaml(run_settings.config_info)

    # os.makedirs(run_settings.run_dir, exist_ok=True)

    # initialize dask-mpi
    dask_mpi.initialize()

    with Client():
        da_out = ds_out['biomass'].compute()
        # if output_file is .zarr, skip compute() and call to_zarr() here
        # could also write to zarr, then read zarr and write to netcdf

    print('\n')
    print('-' * 50)
    print('integration completed')
    print('-' * 50)
    print('\n----\n')

    if run_settings.config_info.get('output_in_2D', False):
        # Rerun map_da_back_to_2D_pop if output_file is not specified in run_settings,
        # or if it is specified in run_settings but does not exist
        print('Converting back to lat/lon grid')
        da_out = feisty.utils.map_da_back_to_2D_pop(
            da_out, run_settings.config_info['forcing']['streams']
        )
        print(da_out)

        # omit .compute and write to zarr?
        if 'output_file' in run_settings.config_info:
            print('\n')
            print('-' * 50)
            print(f'writing output file: {run_settings.config_info["output_file"]}')
            if os.path.isfile(run_settings.config_info['output_file']):
                print(f'{run_settings.config_info["output_file"]} exists, removing now...')
                os.remove(run_settings.config_info['output_file'])
            print('-' * 50)
            encoding = {'biomass': {'_FillValue': 9.969209968386869e36}}
            da_out.to_dataset().to_netcdf(
                run_settings.config_info['output_file'], encoding=encoding
            )

    print(f'Finished at {time.strftime("%H:%M:%S")}')
    print('\n----\n')

    # print('\n')
    # print('-' * 50)
    # print(f'writing restart file: {run_settings.restart_file}')
    # print('-' * 50)

    # run_settings.write_restart_pointer()
    # # TODO: better approach here? Can't write X as multi-index to netCDF
    # ds_out['X'] = np.arange(0, len(ds_out.X))
    # ds_out.isel(time=-1).to_netcdf(run_settings.restart_file)

    # print('\n----\n')

    # # dask_mpi.send_close_signal()
    # if run_settings.resubmit and not run_settings.last_submission:
    #     print('resubmitting')
    #     submit_run(
    #         run_config_file,
    #         run_settings_file,
    #         continue_run=True,
    #     )
    # print('\n')
    # print('-' * 50)
    # print('completed')
    # print('-' * 50)


if __name__ == '__main__':
    main()
