import os

import numpy as np
import xarray as xr
import yaml

from . import testcase
from .core import settings as settings_mod
from .core.interface import feisty_instance_type

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


class simulation(object):
    def __init__(
        self,
        domain_dict,
        forcing,
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

        settings_in : dict
          Settings to overwrite defaults.

        fish_ic_data : numeric, optional
          Initial conditions.

        benthic_prey_ic_data : numeric, optional
          Initial conditions.
        """
        self.domain_dict = domain_dict
        self.forcing = forcing
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
        ]

        self.obj = feisty_instance_type(
            domain_dict=self.domain_dict,
            settings_dict=self.settings_in,
            fish_ic_data=fish_ic_data,
            benthic_prey_ic_data=benthic_prey_ic_data,
        )

    def _forcing_t(self, t):
        return self.forcing.interp(time=t)

    def _init_time_coord(self, nt):
        self.time = xr.DataArray(
            np.arange(1.0, nt + 1.0, 1.0),
            dims=('time'),
            name='time',
            attrs={'long_name': 'time'},
        )

    def _init_output_arrays(self):

        zeros = xr.full_like(self.time, fill_value=0.0)
        ds_diag = zeros * self.obj.tendency_data[self._diagnostic_names]
        ds_prog = zeros * self.obj.get_prognostic().to_dataset()
        self._ds = xr.merge((ds_prog, ds_diag))

    def _post_data(self, n, state_t):
        self._ds.biomass[n, :] = state_t
        for v in self._diagnostic_names:
            self._ds[v][n, :] = self.obj.tendency_data[v]

    @property
    def ds(self):
        """Data comprising the output from a ``feisty`` simulation."""
        return self._ds

    def run(self, nt, file_out=None):
        """Integrate the FEISTY model.

        Parameters
        ----------

        nt : integer
          Number of timesteps to run.
        """

        # get tracer values
        state_t = self.obj.get_prognostic().copy()

        # set up time-coordinate
        self._init_time_coord(nt)

        # initialize memory for output
        self._init_output_arrays()

        # run loop
        for n in range(nt):
            # interpolate forcing
            gcm_data_t = self._forcing_t(self.time[n])

            # compute tendencies
            dfdt = self.obj.compute_tendencies(
                state_t.isel(group=self.obj.prog_ndx_fish),
                state_t.isel(group=self.obj.prog_ndx_benthic_prey),
                gcm_data_t.zooC,
                gcm_data_t.zoo_mort,
                T_pelagic=gcm_data_t.T_pelagic,
                T_bottom=gcm_data_t.T_bottom,
                poc_flux=gcm_data_t.poc_flux_bottom,
            )

            # advance FEISTY state
            state_t[self.obj.prog_ndx_fish, :] = state_t[self.obj.prog_ndx_fish, :] + dfdt * self.dt
            self._post_data(n, state_t)

        self._shutdown(file_out)

    def _shutdown(self, file_out):
        """Close out integration:
        Tasks:
            - write output
        """
        if file_out is not None:
            self._ds.to_netcdf(file_out)


def simulate_testcase(
    domain_name,
    forcing_name,
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
        settings_in,
        fish_ic_data,
        benthic_prey_ic_data,
    )
