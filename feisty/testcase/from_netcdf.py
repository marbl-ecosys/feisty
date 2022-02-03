import numpy as np
import xarray as xr
import yaml

from .idealized import gen_domain_dict


def domain_from_netcdf(forcing_yaml, forcing_key):
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
    return gen_domain_dict(x, bathymetry=bathymetry.data)


def forcing_from_netcdf(domain_dict, forcing_yaml, forcing_key, allow_negative=False):
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
