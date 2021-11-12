import os

import numpy as np
import xarray as xr
import yaml

from .core import settings as settings_mod
from .core.interface import feisty_instance_type


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
            'NX': len(ds_domain.depth_of_seafloor),
            'depth_of_seafloor': ds_domain.depth_of_seafloor,
        }


def _read_fish_init(fich_ic_in):
    pass


def init(
    domain_in,
    settings_in=None,
    fish_ic_in=None,
):
    """Initialize the FEISTY model.

    Parameters
    ----------

    settings_file : string or dict
      File name of a YAML file with FEISTY settings.

    fish_ic_file : string, optional
      NetCDF or Zarr store with initial conditions.

    Returns
    -------

    feisty_instance : feisty_instance_type
      Instantiated `feisty_instance_type`.
    """

    domain_dict = _read_domain(domain_in)
    settings_dict = _read_settings(settings_in)
    fish_ic_data = _read_fish_init(fish_ic_in)

    return feisty_instance_type(
        settings_dict=settings_dict,
        domain_dict=domain_dict,
        fish_ic_data=fish_ic_data,
    )


def compute_tendencies(t, feisty_instance):
    """
    compute tendencies
    """
    pass
    # return feisty_instance.compute_tendencies(
    #     zooplankton_data,
    #     zoo_mortality_data,
    #     T_pelagic=T_pelagic,
    #     T_bottom=T_bottom,
    #     poc_flux=poc_flux,
    # )


def diagnostics_accumulation():
    pass


def step():
    """run the model"""
    pass


def idealized_forcing(nt, nx):
    """generated idealized forcing"""
    time = xr.DataArray(
        np.arange(0, nt),
        dims=('time'),
        name='time',
    )
    X = xr.DataArray(
        np.arange(0, nx),
        dims=('X'),
        name='X',
    )
    return xr.Dataset(
        dict(
            T_pelagic=xr.DataArray(
                np.ones((nt, nx)) * 15.0,
                dims=('time', 'X'),
                name='T_pelagic',
                coords={'time': time, 'X': X},
            ),
            T_bottom=xr.DataArray(
                np.ones((nt, nx)) * 5.0,
                dims=('time', 'X'),
                name='T_bottom',
                coords={'time': time, 'X': X},
            ),
            poc_flux=xr.DataArray(
                np.ones((nt, nx)) * 5.0,
                dims=('time', 'X'),
                name='poc_flux',
                coords={'time': time, 'X': X},
            ),
        )
    )


def get_gcm_forcing_t(ds, time):
    return ds.interp(time=time)


def run(NT):

    # initialize the model
    feisty_instance = init()
    dt = 1.0

    # get tracer values
    fish_biomass = feisty_instance.get_fish_biomass()
    benthic_prey_biomass = feisty_instance.get_benthic_prey_biomass()

    gcm_data = idealized_forcing(nt=10, nx=1)
    time = gcm_data.time

    # run loop
    for n in range(NT):
        # interpolate forcing
        gcm_data_t = get_gcm_forcing_t(gcm_data, time[n])

        # compute tendencies
        dXdt = feisty_instance.compute_tendencies(
            fish_biomass,
            benthic_prey_biomass,
            gcm_data_t.zooplankton_biomass,
            gcm_data_t.zoo_mortality_data,
            T_pelagic=gcm_data_t.T_pelagic,
            T_bottom=gcm_data_t.T_bottom,
            poc_flux=gcm_data_t.poc_flux,
        )

        # advance FEISTY state
        fish_biomass = fish_biomass + dXdt * dt
