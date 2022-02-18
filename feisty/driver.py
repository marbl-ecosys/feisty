import os

import cftime
import numpy as np
import xarray as xr
import yaml

from . import testcase
from .core import settings as settings_mod
from .core.interface import feisty_instance_type

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


def _domain_from_netcdf(forcing_yaml, forcing_key):
    """Read domain information from netcdf file"""
    with open(forcing_yaml) as f:
        forcing_dict = yaml.safe_load(f)[forcing_key]
    ds = xr.open_dataset(forcing_dict['path'])
    if 'dimnames' in forcing_dict:
        for (newdim, olddim) in forcing_dict['dimnames'].items():
            ds = ds.rename({olddim: newdim})
    try:
        bathymetry = ds[forcing_dict['varnames']['bathymetry']]
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


def _forcing_from_netcdf(domain_dict, forcing_yaml, forcing_key, allow_negative=False):
    """Read forcing fields from netcdf file"""
    with open(forcing_yaml) as f:
        forcing_dict = yaml.safe_load(f)[forcing_key]
    ds = xr.open_dataset(forcing_dict['path'])
    if 'dimnames' in forcing_dict:
        for (newdim, olddim) in forcing_dict['dimnames'].items():
            ds = ds.rename({olddim: newdim})
    da_list = []
    for varname in ['T_pelagic', 'T_bottom', 'poc_flux_bottom', 'zooC', 'zoo_mort']:
        try:
            netcdf_varname = forcing_dict['varnames'][varname]
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


class simulation(object):
    def __init__(
        self,
        domain_dict,
        forcing,
        start_date,
        settings_in={},
        fish_ic_data=None,
        benthic_prey_ic_data=None,
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

        # TODO: make this controllable via user input
        self._diagnostic_names = [
            'T_habitat',
            'ingestion_rate',
            'predation_flux',
            'predation_rate',
            'metabolism_rate',
            'mortality_rate',
            'energy_avail_rate',
            'growth_rate',
            'reproduction_rate',
            'recruitment_flux',
            'fish_catch_rate',
            'encounter_rate_link',
            'encounter_rate_total',
            'consumption_rate_max_pred',
            'consumption_rate_link',
        ]

        self.obj = feisty_instance_type(
            domain_dict=self.domain_dict,
            settings_dict=self.settings_in,
            fish_ic_data=fish_ic_data,
            benthic_prey_ic_data=benthic_prey_ic_data,
        )

    def _forcing_t(self, t, ignore_year):
        if ignore_year:
            units = 'days since 0001-01-01 00:00:00'
            interp_time = cftime.num2date(
                cftime.date2num(t, units) % 365, units, calendar=t.calendar
            )
        else:
            interp_time = t
        return self.forcing.interp(time=interp_time)

    def _init_output_arrays(self, nt):
        self.time = xr.cftime_range(start=self.start_date, periods=nt)
        zeros = xr.DataArray(np.zeros(nt), dims=('time'), name='zero')
        ds_diag = zeros * self.obj.tendency_data[self._diagnostic_names]
        ds_prog = zeros * self.obj.get_prognostic().to_dataset()
        self._ds = xr.merge((ds_prog, ds_diag))
        self.ds['time'] = self.time

    def _post_data(self, n, state_t):
        self._ds.biomass[n, :] = state_t
        for v in self._diagnostic_names:
            self._ds[v][n, :] = self.obj.tendency_data[v]

    @property
    def ds(self):
        """Data comprising the output from a ``feisty`` simulation."""
        return self._ds

    def _compute_tendency(self, t, state_t, cyclic_forcing):
        """Return the feisty time tendency."""
        gcm_data_t = self._forcing_t(t, cyclic_forcing)
        return self.obj.compute_tendencies(
            state_t.isel(group=self.obj.prog_ndx_fish),
            state_t.isel(group=self.obj.prog_ndx_benthic_prey),
            gcm_data_t.zooC,
            gcm_data_t.zoo_mort,
            T_pelagic=gcm_data_t.T_pelagic,
            T_bottom=gcm_data_t.T_bottom,
            poc_flux=gcm_data_t.poc_flux_bottom,
        )

    def _solve(self, nt, method, cyclic_forcing):
        """Call a numerical ODE solver to integrate the feisty model in time."""

        state_t = self.obj.get_prognostic().copy()
        self._init_output_arrays(nt)
        if method == 'euler':
            self._solve_foward_euler(nt, state_t, cyclic_forcing)

        elif method in ['Radau', 'RK45']:
            # TODO: make input arguments
            self._solve_scipy(nt, state_t, method, cyclic_forcing)
        else:
            raise ValueError(f'unknown method: {method}')

    def _solve_foward_euler(self, nt, state_t, cyclic_forcing):
        """use forward-euler to solve feisty model"""
        for n in range(nt):
            dsdt = self._compute_tendency(self.time[n], state_t, cyclic_forcing)
            state_t[self.obj.prog_ndx_prognostic, :] = (
                state_t[self.obj.prog_ndx_prognostic, :]
                + dsdt[self.obj.ndx_prognostic, :] * self.dt
            )
            self._post_data(n, state_t)

    def _solve_scipy(self, nt, state_t, method, cyclic_forcing):
        """use a SciPy solver to integrate the model equation."""
        raise NotImplementedError('scipy solvers not implemented')

    def run(self, nt, file_out=None, method='euler', cyclic_forcing=False):
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
        self._solve(nt, method, cyclic_forcing)
        self._shutdown(file_out)

    def _shutdown(self, file_out):
        """Close out integration:
        Tasks:
            - write output
        """
        if file_out is not None:
            self._ds.to_netcdf(file_out)


def config_testcase(
    domain_name,
    forcing_name,
    start_date='0001-01-01',
    settings_in={},
    fish_ic_data=None,
    benthic_prey_ic_data=None,
    domain_kwargs={},
    forcing_kwargs={},
):

    """Return an instance of ``feisty.driver.simulation`` for ``testcase`` data.

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

    sim : feisty.driver.simulation
      An instance of the ``feisty.driver.simulation`` ready for integration.

    Examples
    --------

    Instantiate a ``simulation``::

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

    domain_dict = _test_domain[domain_name](**domain_kwargs)
    forcing = _test_forcing[forcing_name](domain_dict, **forcing_kwargs)

    return simulation(
        domain_dict,
        forcing,
        start_date,
        settings_in,
        fish_ic_data,
        benthic_prey_ic_data,
    )


def config_from_netcdf(
    start_date='0001-01-01',
    settings_in={},
    fish_ic_data=None,
    benthic_prey_ic_data=None,
    domain_kwargs={},
    forcing_kwargs={},
):

    """Return an instance of ``feisty.driver.simulation`` for ``testcase`` data.

    Parameters
    ----------

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

    sim : feisty.driver.simulation
      An instance of the ``feisty.driver.simulation`` ready for integration.

    Examples
    --------

    Instantiate a ``simulation``::

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

    domain_dict = _domain_from_netcdf(**domain_kwargs)
    forcing = _forcing_from_netcdf(domain_dict, **forcing_kwargs)

    return simulation(
        domain_dict,
        forcing,
        start_date,
        settings_in,
        fish_ic_data,
        benthic_prey_ic_data,
    )
