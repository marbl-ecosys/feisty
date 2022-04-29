import numpy as np
import xarray as xr


class domain_params:
    def __init__(
        self,
        _N_points,
        ocean_depth,
    ):
        self._N_points = _N_points
        self.ocean_depth = ocean_depth


def init_module_variables(NX, bathymetry):
    """initialize module variables"""
    assert (
        len(bathymetry) == NX
    ), f'unexpected length for bathymetry; expected {NX}, received {len(bathymetry)}'

    if isinstance(bathymetry, xr.DataArray):
        assert 'X' in bathymetry.coords
    else:
        bathymetry = xr.DataArray(bathymetry, dims=('X'), name='ocean_depth')
        bathymetry = bathymetry.assign_coords({'X': bathymetry})

    return domain_params(NX, bathymetry)


def init_array(mod_params, name=None, constant=None, attrs={}):
    """return initialized array"""
    x = constant if constant is not None else 0.0
    return xr.DataArray(
        np.ones((mod_params._N_points,)) * x,
        dims=('X',),
        coords={'X': mod_params.ocean_depth.X},
        attrs=attrs,
        name=name,
    )


def init_array_2d(mod_params, coord_name, coord_values, name=None, constant=None, attrs={}):
    """return initialized array"""
    coord = xr.DataArray(coord_values, dims=coord_name, name=coord_name)
    return xr.concat(
        [init_array(mod_params, constant=constant, name=name) for f in coord_values],
        dim=coord,
    )
