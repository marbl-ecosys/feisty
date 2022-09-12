import cftime
import xarray as xr


def make_forcing_cyclic(forcing):
    # TODO: better cyclic forcing: first_forcing['forcing_time'] should be the date of forcing.isel(forcing_time=-1) but in the
    #       year before forcing.isel(forcing_time=0); last_forcing['forcing_time'] should be the date of forcing.isel(forcing_time=0)
    #       but in the year after forcing.isel(forcing_time=-1)
    units = 'days since 0001-01-01 00:00:00'
    new_first_forcing = forcing.isel(forcing_time=-1)
    last_forcing_time = forcing['forcing_time'].data[-1]
    new_first_forcing['forcing_time'].data = cftime.num2date(
        cftime.date2num(last_forcing_time, units) % 365 - 365,
        units,
        calendar=last_forcing_time.calendar,
    )
    new_last_forcing = forcing.isel(forcing_time=0)
    first_forcing_time = forcing['forcing_time'].data[0]
    new_last_forcing['forcing_time'].data = cftime.num2date(
        cftime.date2num(first_forcing_time, units) % 365 + 365,
        units,
        calendar=first_forcing_time.calendar,
    )
    return xr.concat(
        [new_first_forcing, forcing, new_last_forcing],
        data_vars='minimal',
        coords='minimal',
        dim='forcing_time',
    )
