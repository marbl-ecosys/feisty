import datetime
import os
import time

import cftime
import numpy as np
import xarray as xr
import yaml
from dask.distributed import wait

from . import testcase
from .core import settings as settings_mod
from .core.interface import feisty_instance_type
from .utils import (
    gen_chunks_dict,
    generate_forcing_ds_from_config,
    generate_ic_ds_for_feisty,
    generate_single_ds_for_feisty,
    generate_template,
    get_forcing_from_config,
    make_forcing_cyclic,
    write_history_file,
    write_restart_file,
)

path_to_here = os.path.dirname(os.path.realpath(__file__))


_test_domain = dict(
    tanh_shelf=testcase.domain_tanh_shelf,
)

_test_forcing = dict(
    cyclic=testcase.forcing_cyclic,
)


def _read_settings(settings_in):
    settings_dict = settings_mod.get_defaults()

    if settings_in is not None:
        if isinstance(settings_in, dict):
            settings_updates = settings_in
        else:
            with open(settings_in) as fid:
                settings_updates = yaml.safe_load(fid)
        settings_dict.update(settings_updates)

    return settings_dict


def _read_fish_init(fich_ic_in):
    pass


class _offline_driver(object):
    def __init__(
        self,
        domain_dict,
        forcing,
        start_date,
        ignore_year_in_forcing=False,
        settings_in={},
        fish_ic_data=None,
        benthic_prey_ic_data=None,
        biomass_init_file=None,
        allow_negative_forcing=False,
        diagnostic_names=[],
        max_output_time_dim=365,
    ):
        """Run an integration with the FEISTY model.

        Parameters
        ----------

        domain_dict : dict
          Dictionary containing ``feisty.domain`` settings.

        forcing : xarray.Dataset
          Forcing data to run the model.

        start_date : str, tuple, or cftime object
          Model year to start simulation.
          If str, format 'YYYY-MM-DD';
          if tuple format (Y, M, D) (all ints).

        settings_in : dict
          Settings to overwrite defaults.

        fish_ic_data : numeric, optional
          Initial conditions.

        benthic_prey_ic_data : numeric, optional
          Initial conditions.
        """
        self._arguments_in = dict(
            domain_dict=domain_dict,
            forcing=forcing,
            start_date=start_date,
            ignore_year_in_forcing=ignore_year_in_forcing,
            settings_in=settings_in,
            fish_ic_data=fish_ic_data,
            benthic_prey_ic_data=benthic_prey_ic_data,
            biomass_init_file=biomass_init_file,
            allow_negative_forcing=allow_negative_forcing,
            diagnostic_names=diagnostic_names,
            max_output_time_dim=max_output_time_dim,
        )
        self.domain_dict = domain_dict
        self.forcing = forcing
        self.allow_negative_forcing = allow_negative_forcing
        self.ignore_year = ignore_year_in_forcing
        date_tuple = None
        if isinstance(start_date, str):
            date_tuple = [int(date_comp) for date_comp in start_date.split('-')]
        elif isinstance(start_date, tuple):
            date_tuple = start_date
        if date_tuple:
            self.start_date = cftime.DatetimeNoLeap(date_tuple[0], date_tuple[1], date_tuple[2])
        else:
            self.start_date = start_date
        self.settings_in = settings_in
        self.dt = 1.0  # day
        self._max_output_time_dim = max_output_time_dim

        # TODO: make this controllable via user input
        self._diagnostic_names = diagnostic_names

        self.obj = feisty_instance_type(
            domain_dict=self.domain_dict,
            settings_dict=self.settings_in,
            fish_ic_data=fish_ic_data,
            benthic_prey_ic_data=benthic_prey_ic_data,
            biomass_init_file=biomass_init_file,
        )

    def _init_output_arrays(self, nt):
        """Create self._ds_list, a list of datasets containing no more than
        self._max_output_time_dim time levels
        """
        self._ds_list = []
        start_dates = [self.start_date]
        time_levs_per_ds = [np.min([nt, self._max_output_time_dim])]
        total_days = nt - time_levs_per_ds[0]
        while total_days != 0:
            start_dates.append(start_dates[-1] + datetime.timedelta(int(time_levs_per_ds[-1])))
            time_levs_per_ds.append(np.min([self._max_output_time_dim, total_days]))
            total_days = total_days - time_levs_per_ds[-1]
        ind_cnt = 0
        for ind in time_levs_per_ds:
            ind_cnt += ind
        forcing_time = []
        for nt_loc, start_date in zip(time_levs_per_ds, start_dates):
            time = xr.cftime_range(start=start_date, periods=nt_loc, calendar='noleap')
            if self.ignore_year:
                units = 'days since 0001-01-01 00:00:00'
                forcing_time.append(
                    np.array(
                        [
                            cftime.num2date(
                                cftime.date2num(t, units) % 365, units, calendar=t.calendar
                            )
                            for t in time
                        ]
                    )
                )
            else:
                forcing_time_loc = np.where(
                    time > self.forcing['forcing_time'].data[0],
                    time,
                    self.forcing['forcing_time'].data[0],
                )
                forcing_time_loc = np.where(
                    forcing_time_loc < self.forcing['forcing_time'].data[-1],
                    forcing_time_loc,
                    self.forcing['forcing_time'].data[-1],
                )
                forcing_time.append(forcing_time_loc)
            zeros = xr.DataArray(np.zeros(nt_loc), dims=('time'), name='zero')
            ds_prog = zeros * self.state_t.to_dataset()
            if self._diagnostic_names:
                ds_diag = zeros * self.obj.tendency_data[self._diagnostic_names]
                ds_diag = ds_diag.assign_coords({'X': ds_prog.X.data})
                self._ds_list.append(xr.merge((ds_prog, ds_diag)))
            else:
                self._ds_list.append(ds_prog)
            self._ds_list[-1]['time'] = time
            self._ds_list[-1] = self._ds_list[-1].assign_coords({'X': self.forcing.X.data})
        self._forcing_time = np.concatenate(forcing_time)

    def _post_data(self, n):
        # Which file do we write to?
        ds_ind = n // self._max_output_time_dim
        data_ind = n % self._max_output_time_dim
        if n > 0 and data_ind == 0:
            print(f'Starting a new output dataset for timestep {n} ({time.strftime("%H:%M:%S")})')
        self._ds_list[ds_ind].biomass.data[data_ind, :, :] = self.state_t.data
        for v in self._diagnostic_names:
            print(f'{v}: {np.shape(self.obj.tendency_data[v].data)}')
            if v == 'forcings':
                pass
            else:
                self._ds_list[ds_ind][v].data[data_ind, :] = self.obj.tendency_data[v].data

    # @property
    # def ds(self):
    #     """Data comprising the output from ``feisty``."""
    #     return self._ds_list

    def _compute_tendency(self, forcing_t):
        """Return the feisty time tendency."""
        gcm_data_t = self.forcing.interp(forcing_time=forcing_t, assume_sorted=True)
        if not self.allow_negative_forcing:
            gcm_data_t['poc_flux_bottom'].data = np.maximum(gcm_data_t['poc_flux_bottom'].data, 0)
            gcm_data_t['zooC'].data = np.maximum(gcm_data_t['zooC'].data, 0)
            gcm_data_t['zoo_mort'].data = np.maximum(gcm_data_t['zoo_mort'].data, 0)
        return self.obj.compute_tendencies(
            self.state_t.data[self.obj.prog_ndx_fish, :],
            self.state_t.data[self.obj.prog_ndx_benthic_prey, :],
            gcm_data_t.zooC.data,
            gcm_data_t.zoo_mort.data,
            T_pelagic=gcm_data_t.T_pelagic.data,
            T_bottom=gcm_data_t.T_bottom.data,
            poc_flux=gcm_data_t.poc_flux_bottom.data,
        )

    def _solve(self, nt, method):
        """Call a numerical ODE solver to integrate the feisty model in time."""

        if method == 'euler':
            self._solve_foward_euler(nt)

        elif method in ['Radau', 'RK45']:
            # TODO: make input arguments
            self._solve_scipy(nt, method)
        else:
            raise ValueError(f'unknown method: {method}')

    def _solve_foward_euler(self, nt):
        """use forward-euler to solve feisty model"""
        print(f'Integrating {nt} steps (starting at {time.strftime("%H:%M:%S")})')
        for n in range(nt):
            dsdt = self._compute_tendency(self._forcing_time[n])
            self.state_t.data[self.obj.prog_ndx_prognostic, :] = (
                self.state_t[self.obj.prog_ndx_prognostic, :].data
                + dsdt.data[self.obj.ndx_prognostic, :] * self.dt
            )
            self._post_data(n)

    def _solve_scipy(self, nt, method):
        """use a SciPy solver to integrate the model equation."""
        raise NotImplementedError('scipy solvers not implemented')

    def run(self, nt, file_out=None, method='euler'):
        """Integrate the FEISTY model.

        Parameters
        ----------

        nt : integer
          Number of timesteps to run.

        file_out : string
          File name to write model output data.

        method : string
          Method of solving feisty equations. Options: ['euler', 'Radau', 'RK45'].

          .. note::
             Only ``method='euler'`` is supported currently.

        """
        print(f'Starting run() at {time.strftime("%H:%M:%S")}')
        self.state_t = self.obj.get_prognostic().copy().assign_coords({'X': self.forcing.X.data})
        self._init_output_arrays(nt)
        self._solve(nt, method)
        self._shutdown(file_out)

    def run_parallel(self, nt, file_out=None, method='euler'):
        """Integrate the FEISTY model.

        Parameters
        ----------

        nt : integer
          Number of timesteps to run.

        file_out : string
          File name to write model output data.

        method : string
          Method of solving feisty equations. Options: ['euler', 'Radau', 'RK45'].

          .. note::
             Only ``method='euler'`` is supported currently.

        """
        print(f'Starting run() at {time.strftime("%H:%M:%S")}')
        self.state_t = self.obj.get_prognostic().copy().assign_coords({'X': self.forcing.X.data})
        self._init_output_arrays(nt)
        forcing_time_offset = 0
        for ds_ind in range(len(self._ds_list)):
            # Parallelize?
            self.template = self._ds_list[ds_ind]

            nt_loc = len(self._ds_list[ds_ind].time)
            t0 = self._ds_list[ds_ind].time[0].data
            print(f'Integrating {nt_loc} steps from {t0} (starting at {time.strftime("%H:%M:%S")})')
            new_ds = None
            forcing_time_offset = forcing_time_offset + nt_loc
            self._ds_list[ds_ind] = new_ds.compute()
            self.state_t.data = new_ds.isel(time=-1).biomass.data
            # self._solve(nt_loc, method)
        self._shutdown(file_out)

    def gen_ds(self):
        if len(self._ds_list) == 1:
            self.ds = self._ds_list[0].copy(deep=True)
        else:
            self.ds = xr.concat(
                self._ds_list,
                dim='time',
                data_vars='minimal',
                coords='minimal',
                compat='override',
            )

    def _shutdown(self, file_out):
        """Close out integration:
        Tasks:
            - write output
        """
        if file_out is not None:
            print(f'Finished _solve and calling _shutdown() at {time.strftime("%H:%M:%S")}')
            self.gen_ds()
            self.ds.to_netcdf(file_out)
            print(f'Finished _shutdown() at {time.strftime("%H:%M:%S")}')
        else:
            print(f'Finished _solve at {time.strftime("%H:%M:%S")}')


# PUBLIC FUNCTIONS
def config_and_run_testcase(
    domain_name,
    forcing_name,
    start_date='0001-01-01',
    end_date='0001-01-01',
    settings_in={},
    fish_ic_data=1e-5,
    benthic_prey_ic_data=2e-3,
    domain_kwargs={},
    forcing_kwargs={},
    diagnostic_names=[],
    max_output_time_dim=365,
    method='euler',
):

    """Return an instance of ``feisty.driver.offline_driver`` for ``testcase`` data.

    Parameters
    ----------

    domain_name : string
      Name of domain testcase.

    forcing_name : string
      Name of forcing testcase.

    start_date : str (or tuple or cftime object)
      Model year to start simulation.

    settings_in : dict
      Settings to overwrite defaults.

    fish_ic_data : numeric, array_like
      Initial conditions.

    benthic_prey_ic_data : numeric, array_like
      Initial conditions.

    domain_kwargs : dict
      Keyword arguments to pass to domain generation function.

    forcing_kwargs : dict
      Keyword arguments to pass to forcing generation function.

    Returns
    -------

    sim : feisty.driver.offline_driver
      An instance of the ``feisty.driver.offline_driver`` ready for integration.

    Examples
    --------

    Instantiate a ``offline_driver``::

      >>> testcase = feisty.driver.simulate_testcase("tanh_shelf", "cyclic")

    Integrate the model for 365 days::

       >>> testcase.run(365)

    Access the output::

      >>> testcase.ds.info()
      xarray.Dataset {
      dimensions:
              X = 22 ;
              group = 9 ;
              time = 365 ;
              fish = 8 ;
      variables:
              float64 X(X) ;
              <U12 group(group) ;
              float64 biomass(time, group, X) ;
              <U2 fish(fish) ;
              float64 T_habitat(time, fish, X) ;
              float64 ingestion_rate(time, fish, X) ;
              float64 predation_flux(time, fish, X) ;
              float64 predation_rate(time, fish, X) ;
              float64 metabolism_rate(time, fish, X) ;
              float64 mortality_rate(time, fish, X) ;
              float64 energy_avail_rate(time, fish, X) ;
              float64 growth_rate(time, fish, X) ;
              float64 reproduction_rate(time, fish, X) ;
              float64 recruitment_flux(time, fish, X) ;
              float64 fish_catch_rate(time, fish, X) ;
      // global attributes:
        }
    """

    assert domain_name in _test_domain
    assert forcing_name in _test_forcing

    # Set up forcing / initial condition dictionary
    domain_dict = _test_domain[domain_name](**domain_kwargs)
    ds = _test_forcing[forcing_name](domain_dict, **forcing_kwargs)
    ds['bathymetry'] = domain_dict['bathymetry'].assign_coords(X=ds.X.data)
    ds_ic = generate_ic_ds_for_feisty(
        ds,
        ic_file=None,
        fish_ic=fish_ic_data,
        benthic_prey_ic=benthic_prey_ic_data,
    )
    for var in ['fish_ic', 'bent_ic']:
        ds[var] = ds_ic[var]

    # Set up template for map_blocks
    template = generate_template(
        ds=ds,
        start_date=start_date,
        end_date=end_date,
        diagnostic_names=diagnostic_names,
    )

    return xr.map_blocks(
        config_and_run_from_dataset,
        ds,
        args=(
            ds_ic,
            len(template['time']),
            start_date,
            True,  # ignore_year_in_forcing is always true for test case
            settings_in,
            diagnostic_names,
            max_output_time_dim,
            method,
        ),
        template=template,
    ).compute()


def config_and_run_from_netcdf(
    input_yaml,
    input_key,
    start_date='0001-01-01',
    end_date='0001-01-01',
    ignore_year_in_forcing=False,
    settings_in={},
    fish_ic_data=1e-5,
    benthic_prey_ic_data=2e-3,
    diagnostic_names=[],
    max_output_time_dim=365,
    num_chunks=1,
    method='euler',
):

    """Return an instance of ``feisty.driver.offline_driver`` configured from a YAML file

    Parameters
    ----------

    input_yaml : str
      File name of a YAML file containing configuration information

    input_key : str
      Top-level key in input_yaml specifying which run is being configured

    start_date : str (or tuple or cftime object)
      Model year to start simulation.

    settings_in : dict
      Settings to overwrite defaults.

    fish_ic_data : numeric, array_like
      Initial conditions.

    benthic_prey_ic_data : numeric, array_like
      Initial conditions.

    Returns
    -------

    sim : feisty.driver.offline_driver
      An instance of the ``feisty.driver.offline_driver`` ready for integration.

    # Examples
    # --------

    # Instantiate a ``offline_driver``::

    #   >>> testcase = feisty.driver.simulate_testcase("tanh_shelf", "cyclic")

    # Integrate the model for 365 days::

    #    >>> testcase.run(365)

    # Access the output::

    #   >>> testcase.ds.info()
    #   xarray.Dataset {
    #   dimensions:
    #           X = 22 ;
    #           group = 9 ;
    #           time = 365 ;
    #           fish = 8 ;
    #   variables:
    #           float64 X(X) ;
    #           <U12 group(group) ;
    #           float64 biomass(time, group, X) ;
    #           <U2 fish(fish) ;
    #           float64 T_habitat(time, fish, X) ;
    #           float64 ingestion_rate(time, fish, X) ;
    #           float64 predation_flux(time, fish, X) ;
    #           float64 predation_rate(time, fish, X) ;
    #           float64 metabolism_rate(time, fish, X) ;
    #           float64 mortality_rate(time, fish, X) ;
    #           float64 energy_avail_rate(time, fish, X) ;
    #           float64 growth_rate(time, fish, X) ;
    #           float64 reproduction_rate(time, fish, X) ;
    #           float64 recruitment_flux(time, fish, X) ;
    #           float64 fish_catch_rate(time, fish, X) ;
    #   // global attributes:
    #     }
    """
    # Determine location of initial conditions
    # "constant" => 1e-5 for fish, 2e-3 for benthic prey
    with open(input_yaml) as f:
        input_dict = yaml.safe_load(f)[input_key]

    forcing_rename = dict()
    forcing_rename['time'] = 'forcing_time'
    for input_key in ['dimnames', 'varnames']:
        if input_key in input_dict:
            for key, value in input_dict[input_key].items():
                forcing_rename[value] = key

    ds = generate_single_ds_for_feisty(
        num_chunks=num_chunks,
        forcing_file=input_dict['path'],
        forcing_rename=forcing_rename,
    )
    if ignore_year_in_forcing:
        ds = make_forcing_cyclic(ds, cyclic_year=1)

    ds_ic = generate_ic_ds_for_feisty(
        ds,
        ic_file=input_dict.get('biomass_init_file', None),
        fish_ic=input_dict.get('fish_biomass_ic', fish_ic_data),
        benthic_prey_ic=input_dict.get('benthic_prey_biomass_ic', benthic_prey_ic_data),
    )

    template = generate_template(
        ds=ds,
        start_date=start_date,
        end_date=end_date,
        diagnostic_names=diagnostic_names,
    )

    return xr.map_blocks(
        config_and_run_from_dataset,
        ds,
        args=(
            ds_ic,
            len(template['time']),
            start_date,
            ignore_year_in_forcing,
            settings_in,
            diagnostic_names,
            max_output_time_dim,
            method,
        ),
        template=template,
    ).compute()


def config_and_run_from_yaml(input_dict, settings_in={}, ds=None):

    """Return an instance of ``feisty.driver.offline_driver`` configured from a YAML file.
       This should eventually replace config_and_run_from_netcdf() as it is a better YAML schema.

    Parameters
    ----------

    input_dict : dict
      Configuration of this FEISTY case

    settings_in : dict
      Settings to overwrite defaults.

    Returns
    -------

    sim : feisty.driver.offline_driver
      An instance of the ``feisty.driver.offline_driver`` ready for integration.

    # Examples
    # --------

    # Instantiate a ``offline_driver``::

    #   >>> testcase = feisty.driver.simulate_testcase("tanh_shelf", "cyclic")

    # Integrate the model for 365 days::

    #    >>> testcase.run(365)

    # Access the output::

    #   >>> testcase.ds.info()
    #   xarray.Dataset {
    #   dimensions:
    #           X = 22 ;
    #           group = 9 ;
    #           time = 365 ;
    #           fish = 8 ;
    #   variables:
    #           float64 X(X) ;
    #           <U12 group(group) ;
    #           float64 biomass(time, group, X) ;
    #           <U2 fish(fish) ;
    #           float64 T_habitat(time, fish, X) ;
    #           float64 ingestion_rate(time, fish, X) ;
    #           float64 predation_flux(time, fish, X) ;
    #           float64 predation_rate(time, fish, X) ;
    #           float64 metabolism_rate(time, fish, X) ;
    #           float64 mortality_rate(time, fish, X) ;
    #           float64 energy_avail_rate(time, fish, X) ;
    #           float64 growth_rate(time, fish, X) ;
    #           float64 reproduction_rate(time, fish, X) ;
    #           float64 recruitment_flux(time, fish, X) ;
    #           float64 fish_catch_rate(time, fish, X) ;
    #   // global attributes:
    #     }
    """

    num_workers = input_dict.get('num_workers', 1)
    chunks = input_dict.get('chunks') if num_workers > 1 else None

    start_date = input_dict['start_date']
    end_date = input_dict['end_date']
    POP_units = input_dict['forcing'].get('POP_units', False)
    ignore_year_in_forcing = input_dict['forcing'].get('use_cyclic_forcing', False)

    diagnostic_names = input_dict.get('output', {}).get('diagnostic_names', [])
    method = input_dict.get('method', 'euler')
    max_output_time_dim = input_dict.get('max_output_time_dim', 365)

    feisty_forcing = get_forcing_from_config(input_dict)

    # include date for zarr
    debug_outdir = '/glade/derecho/scratch/akenney/final.zarr'

    if ds is None:
        ds = generate_forcing_ds_from_config(
            feisty_forcing, chunks, POP_units, debug_outdir=debug_outdir
        )

        if ignore_year_in_forcing:
            ds = make_forcing_cyclic(ds, input_dict['forcing'].get('cyclic_year', 1))

    if 'initial_conditions' in input_dict:
        ic_root = input_dict['initial_conditions'].get('root_dir', '.')
        ic_file = f"{ic_root}/{input_dict['initial_conditions']['ic_file']}"
    else:
        ic_file = None
    ds_ic = generate_ic_ds_for_feisty(
        ds,
        ic_file=ic_file,
        chunks=chunks,
    )

    template = generate_template(
        ds=ds,
        start_date=start_date,
        end_date=end_date,
        diagnostic_names=diagnostic_names,
    )

    ds_out = xr.map_blocks(
        config_and_run_from_dataset,
        ds,
        args=(
            ds_ic,
            len(template['time']),
            start_date,
            ignore_year_in_forcing,
            settings_in,
            diagnostic_names,
            max_output_time_dim,
            method,
        ),
        template=template,
    ).persist()
    wait(ds_out)

    # Output according to YAML
    if 'output' in input_dict:
        overwrite = input_dict['output'].get('overwrite', False)
        if 'hist_file' in input_dict['output']:
            hist_dir = input_dict['output'].get('hist_dir', '.')
            write_history_file(
                ds_out,
                os.path.join(hist_dir, input_dict['output']['hist_file']),
                overwrite=overwrite,
            )
        if 'rest_file' in input_dict['output']:
            rest_dir = input_dict['output'].get('rest_dir', '.')
            write_restart_file(
                ds_out,
                os.path.join(rest_dir, input_dict['output']['rest_file']),
                overwrite=overwrite,
            )

    return ds_out


def config_and_run_from_dataset(
    ds,
    ds_ic,
    nstep,
    start_date='0001-01-01',
    ignore_year_in_forcing=False,
    settings_in={},
    diagnostic_names=[],
    max_output_time_dim=365,
    method='euler',
):
    """This routine
    1. creates an _offline_driver object
    2. calls run(nstep)
    3. calls gen_ds()
    4. returns the resulting ds
    """
    stacked = False
    if 'X' not in ds:
        stacked = True
        ds = ds.stack(X=('nlat', 'nlon'))
        ds_ic = ds_ic.stack(X=('nlat', 'nlon'))
        # # drop land points
        # ocean_mask = ds['bathymetry'] > 0
        # ds = ds.where(ocean_mask, drop=True)
        # ds_ic = ds_ic.where(ocean_mask, drop=True)

    domain_dict = dict()
    domain_dict['bathymetry'] = ds['bathymetry']
    domain_dict['NX'] = len(ds['X'])

    forcing = ds[['T_pelagic', 'T_bottom', 'poc_flux_bottom', 'zooC', 'zoo_mort']]

    fish_ic_data = ds_ic['fish_ic']
    benthic_prey_ic_data = ds_ic['bent_ic']

    feisty_driver = _offline_driver(
        domain_dict,
        forcing,
        start_date,
        ignore_year_in_forcing,
        settings_in,
        fish_ic_data,
        benthic_prey_ic_data,
        diagnostic_names=diagnostic_names,
        max_output_time_dim=max_output_time_dim,
    )

    feisty_driver.run(nstep, method=method)
    feisty_driver.gen_ds()

    # apply mask to output variables listed in diagnostic_names
    if diagnostic_names:
        mask = np.isfinite(feisty_driver.ds['biomass'].isel(time=0, group=0).data)
        for variable in diagnostic_names:
            feisty_driver.ds[variable].data = np.where(
                mask, feisty_driver.ds[variable].data, np.nan
            )

    if stacked:
        # Need to get back to nlat, nlon dimension from X MultiIndex
        feisty_driver.ds = feisty_driver.ds.drop(['X'])
        feisty_driver.ds = feisty_driver.ds.assign_coords(
            {
                'nlat': xr.DataArray(ds['nlat'].data, dims='X'),
                'nlon': xr.DataArray(ds['nlon'].data, dims='X'),
            }
        )
        feisty_driver.ds['X'] = ds.indexes['X']
        feisty_driver.ds = feisty_driver.ds.unstack()

    return feisty_driver.ds
