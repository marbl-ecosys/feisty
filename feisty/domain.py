import numpy as np
import xarray as xr

_N_points = None
ocean_depth = None


def init_module_variables(NX, depth_of_seafloor):
    """initialize module variables"""
    global _N_points
    global ocean_depth

    assert (
        len(depth_of_seafloor) == NX
    ), f'unexpected length for depth_of_seafloor; expected {NX}, received {len(depth_of_seafloor)}'

    _N_points = NX
    ocean_depth = depth_of_seafloor


def init_array(name=None, constant=None, attrs={}):
    """return initialized array"""
    x = constant if constant is not None else 0.0
    return xr.DataArray(
        np.ones((_N_points,)) * x,
        dims=('X',),
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
