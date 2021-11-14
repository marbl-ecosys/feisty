import numpy as np
import xarray as xr

_N_points = None
ocean_depth = None


def init_module_variables(NX, bathymetry):
    """initialize module variables"""
    global _N_points
    global ocean_depth

    assert (
        len(bathymetry) == NX
    ), f'unexpected length for bathymetry; expected {NX}, received {len(bathymetry)}'

    _N_points = NX
    if isinstance(bathymetry, xr.DataArray):
        assert 'X' in bathymetry.coords
    else:
        bathymetry = xr.DataArray(bathymetry, dims=('X'), name='ocean_depth')
        bathymetry = bathymetry.assign_coords({'X': bathymetry})

    ocean_depth = bathymetry


def init_array(name=None, constant=None, attrs={}):
    """return initialized array"""
    x = constant if constant is not None else 0.0
    return xr.DataArray(
        np.ones((_N_points,)) * x,
        dims=('X',),
        coords={'X': ocean_depth.X},
        attrs=attrs,
        name=name,
    )


def init_array_2d(coord_name, coord_values, name=None, constant=None, attrs={}):
    """return initialized array"""
    coord = xr.DataArray(coord_values, dims=coord_name, name=coord_name)
    return xr.concat(
        [init_array(constant=constant, name=name) for f in coord_values],
        dim=coord,
    )
