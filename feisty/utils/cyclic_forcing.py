import cftime
import xarray as xr


def make_forcing_cyclic(forcing):
    # TODO: better cyclic forcing: first_forcing['forcing_time'] should be the date of forcing.isel(forcing_time=-1) but in the
    #       year before forcing.isel(forcing_time=0); last_forcing['forcing_time'] should be the date of forcing.isel(forcing_time=0)
    #       but in the year after forcing.isel(forcing_time=-1)
    units = 'days since 0001-01-01 00:00:00'
    first_forcing = forcing.isel(forcing_time=-1)
    old_data = forcing['forcing_time'].data[-1]
    first_forcing['forcing_time'].data = cftime.num2date(
        cftime.date2num(old_data, units) % 365 - 365, units, calendar=old_data.calendar
    )
    last_forcing = forcing.isel(forcing_time=0)
    old_data = forcing['forcing_time'].data[0]
    last_forcing['forcing_time'].data = cftime.num2date(
        cftime.date2num(old_data, units) % 365 + 365, units, calendar=old_data.calendar
    )
    return xr.concat([first_forcing, forcing, last_forcing], dim='forcing_time')
