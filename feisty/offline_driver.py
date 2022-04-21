import datetime
import os
import time

import cftime
import numpy as np
import xarray as xr
import yaml

from . import testcase
from .core import settings as settings_mod
from .core.interface import feisty_instance_type
from .utils import make_forcing_cyclic

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


def _read_domain(domain_in, test_case=None):
    with xr.open_dataset(domain_in) as ds_domain:
        return {
            'NX': len(ds_domain.bathymetry),
            'bathymetry': ds_domain.bathymetry,
        }


def _read_fish_init(fich_ic_in):
    pass


def _domain_from_netcdf(input_dict):
    """Read domain information from netcdf file"""
    ds = xr.open_dataset(input_dict['path'])
    if 'dimnames' in input_dict:
        for (newdim, olddim) in input_dict['dimnames'].items():
            ds = ds.rename({olddim: newdim})
    try:
        bathymetry = ds[input_dict['varnames']['bathymetry']]
    except:
        bathymetry = ds['bathymetry']
    x = np.arange(bathymetry.size).reshape(bathymetry.shape)
    return dict(
        bathymetry=xr.DataArray(
            bathymetry.data,
            dims=('X'),
            name='bathymetry',
            attrs={'long_name': 'depth', 'units': 'm'},
            coords={'X': x},
        ),
        NX=len(bathymetry.data),
    )


def _forcing_from_netcdf(input_dict):
    """Read forcing fields from netcdf file"""
    allow_negative = input_dict.get('allow_negative', False)
    ds = xr.open_dataset(input_dict['path'])
    if 'dimnames' in input_dict:
        for (newdim, olddim) in input_dict['dimnames'].items():
            ds = ds.rename({olddim: newdim})
    da_list = []
    for varname in ['T_pelagic', 'T_bottom', 'poc_flux_bottom', 'zooC', 'zoo_mort']:
        try:
            netcdf_varname = input_dict['varnames'][varname]
            da_list.append(ds[netcdf_varname].rename(varname))
        except:
            da_list.append(ds[varname])
    ds = xr.merge(da_list)
    if not allow_negative:
        for varname in ['poc_flux_bottom', 'zooC', 'zoo_mort']:
            ds[varname].data = np.where(ds[varname].data > 0, ds[varname].data, 0)
    if 'zooplankton' not in ds.dims:
        ds['zooC'] = ds['zooC'].expand_dims('zooplankton')
        ds['zoo_mort'] = ds['zoo_mort'].expand_dims('zooplankton')
        ds['zooplankton'] = xr.DataArray(['Zoo'], dims='zooplankton')

    return ds


class offline_driver(object):
    def __init__(
        self,
        domain_dict,
        forcing,
        start_date,
        ignore_year_in_forcing=False,
        settings_in={},
        fish_ic_data=None,
        benthic_prey_ic_data=None,
        biomass_init='constant',
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
            biomass_init=biomass_init,
        )

    def _init_output_arrays(self, nt):
        """Create self._ds, a list of datasets containing no more than
        self._max_output_time_dim time levels
        """
        self._ds = []
        start_dates = [self.start_date]
        days_per_year = [np.min([nt, self._max_output_time_dim])]
        total_days = nt - days_per_year[0]
        while total_days != 0:
            start_dates.append(start_dates[-1] + datetime.timedelta(int(days_per_year[-1])))
            days_per_year.append(np.min([self._max_output_time_dim, total_days]))
            total_days = total_days - days_per_year[-1]
        self._ds_ind = np.zeros(nt, dtype='int')
        ind_cnt = 0
        for ind in days_per_year:
            ind_cnt += ind
            self._ds_ind[ind_cnt:] += 1
        forcing_time = []
        for nt_loc, start_date in zip(days_per_year, start_dates):
            self.time = xr.cftime_range(start=start_date, periods=nt_loc, calendar='noleap')
            if self.ignore_year:
                units = 'days since 0001-01-01 00:00:00'
                forcing_time.append(
                    np.array(
                        [
                            cftime.num2date(
                                cftime.date2num(t, units) % 365, units, calendar=t.calendar
                            )
                            for t in self.time
                        ]
                    )
                )
            else:
                forcing_time_loc = np.where(
                    self.time > self.forcing['time'].data[0],
                    self.time,
                    self.forcing['time'].data[0],
                )
                forcing_time_loc = np.where(
                    forcing_time_loc < self.forcing['time'].data[-1],
                    forcing_time_loc,
                    self.forcing['time'].data[-1],
                )
                forcing_time.append(forcing_time_loc)
            zeros = xr.DataArray(np.zeros(nt_loc), dims=('time'), name='zero')
            ds_diag = zeros * self.obj.tendency_data[self._diagnostic_names]
            ds_prog = zeros * self.obj.get_prognostic().to_dataset()
            self._ds.append(xr.merge((ds_prog, ds_diag)))
            self._ds[-1]['time'] = self.time
        self._forcing_time = np.concatenate(forcing_time)

    def _post_data(self, n, state_t):
        # Which file do we write to?
        ds_ind = self._ds_ind[n]
        data_ind = n % self._max_output_time_dim
        self._ds[ds_ind].biomass.data[data_ind, :, :] = state_t.data
        for v in self._diagnostic_names:
            self._ds[ds_ind][v].data[data_ind, :] = self.obj.tendency_data[v].data

    # @property
    # def ds(self):
    #     """Data comprising the output from ``feisty``."""
    #     return self._ds

    def _compute_tendency(self, forcing_t, state_t):
        """Return the feisty time tendency."""
        gcm_data_t = self.forcing.interp(time=forcing_t, assume_sorted=True)
        if not self.allow_negative_forcing:
            gcm_data_t['poc_flux_bottom'].data = np.maximum(gcm_data_t['poc_flux_bottom'].data, 0)
            gcm_data_t['zooC'].data = np.maximum(gcm_data_t['zooC'].data, 0)
            gcm_data_t['zoo_mort'].data = np.maximum(gcm_data_t['zoo_mort'].data, 0)
        return self.obj.compute_tendencies(
            state_t.data[self.obj.prog_ndx_fish, :],
            state_t.data[self.obj.prog_ndx_benthic_prey, :],
            gcm_data_t.zooC.data,
            gcm_data_t.zoo_mort.data,
            T_pelagic=gcm_data_t.T_pelagic.data,
            T_bottom=gcm_data_t.T_bottom.data,
            poc_flux=gcm_data_t.poc_flux_bottom.data,
        )

    def _solve(self, nt, method):
        """Call a numerical ODE solver to integrate the feisty model in time."""

        state_t = self.obj.get_prognostic().copy()
        self._init_output_arrays(nt)
        if method == 'euler':
            self._solve_foward_euler(nt, state_t)

        elif method in ['Radau', 'RK45']:
            # TODO: make input arguments
            self._solve_scipy(nt, state_t, method)
        else:
            raise ValueError(f'unknown method: {method}')

    def _solve_foward_euler(self, nt, state_t):
        """use forward-euler to solve feisty model"""
        for n in range(nt):
            if n % 365 == 0:
                print(f'Starting year {n//365+1} integration at {time.strftime("%H:%M:%S")}')
            dsdt = self._compute_tendency(self._forcing_time[n], state_t)
            state_t.data[self.obj.prog_ndx_prognostic, :] = (
                state_t[self.obj.prog_ndx_prognostic, :].data
                + dsdt.data[self.obj.ndx_prognostic, :] * self.dt
            )
            self._post_data(n, state_t)

    def _solve_scipy(self, nt, state_t, method):
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
        print(f'Calling _solve at {time.strftime("%H:%M:%S")}')
        self._solve(nt, method)
        self._shutdown(file_out)

    def gen_ds(self):
        if len(self._ds) == 1:
            self.ds = self._ds[0].copy(deep=True)
        else:
            self.ds = xr.concat(
                self._ds,
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

    def create_ic_file_from_final_state(self, ic_file, overwrite=False):
        if os.path.isfile(ic_file):
            if not overwrite:
                print(f'{ic_file} exists; set overwrite=True to replace')
                return
            print(f'Removing {ic_file} before writing new copy')
            os.remove(ic_file)

        fish_ic = (
            self._ds[-1]
            .isel(time=-1, group=self.obj.prog_ndx_fish)
            .drop(['X', 'time', 'group'])
            .biomass.rename({'group': 'nfish'})
            .to_dataset(name='fish_ic')
        )
        bent_ic = (
            self._ds[-1]
            .isel(time=-1, group=self.obj.prog_ndx_benthic_prey)
            .drop(['X', 'time', 'group'])
            .biomass.rename({'group': 'nb'})
            .to_dataset(name='bent_ic')
        )
        new_ic = xr.merge([fish_ic, bent_ic])
        new_ic.to_netcdf(ic_file, encoding={v: {'_FillValue': None} for v in new_ic.variables})


def config_testcase(
    domain_name,
    forcing_name,
    start_date='0001-01-01',
    settings_in={},
    fish_ic_data=None,
    benthic_prey_ic_data=None,
    domain_kwargs={},
    forcing_kwargs={},
    diagnostic_names=[],
    max_output_time_dim=365,
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

    # Set up domain_dict and forcing
    domain_dict = _test_domain[domain_name](**domain_kwargs)
    forcing = _test_forcing[forcing_name](domain_dict, **forcing_kwargs)

    # Enable cyclic forcing for certain tests
    ignore_year_in_forcing = forcing_name in ['cyclic']
    if ignore_year_in_forcing:
        forcing = make_forcing_cyclic(forcing)

    return offline_driver(
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


def config_from_netcdf(
    input_yaml,
    input_key,
    start_date='0001-01-01',
    ignore_year_in_forcing=False,
    settings_in={},
    fish_ic_data=None,
    benthic_prey_ic_data=None,
    diagnostic_names=[],
    max_output_time_dim=365,
):

    """Return an instance of ``feisty.driver.offline_driver`` for ``testcase`` data.

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
    # Determine location of initial conditions
    # "constant" => 1e-5 for fish, 2e-3 for benthic prey
    with open(input_yaml) as f:
        input_dict = yaml.safe_load(f)[input_key]

    domain_dict = _domain_from_netcdf(input_dict)
    forcing = _forcing_from_netcdf(input_dict)
    if ignore_year_in_forcing:
        forcing = make_forcing_cyclic(forcing)

    return offline_driver(
        domain_dict,
        forcing,
        start_date,
        ignore_year_in_forcing,
        settings_in,
        fish_ic_data,
        benthic_prey_ic_data,
        input_dict.get('biomass_init', 'constant'),
        input_dict.get('allow_negative', False),
        diagnostic_names=diagnostic_names,
        max_output_time_dim=max_output_time_dim,
    )
