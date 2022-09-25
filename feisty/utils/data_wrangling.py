import time
from dataclasses import dataclass

import numpy as np
import xarray as xr


def generate_ic_ds_for_feisty(
    X, ic_file, num_chunks, fish_ic=1e-5, benthic_prey_ic=2e-3, ic_rename={}
):
    if ic_file is None:
        nX = len(X)
        ds_ic = xr.Dataset(
            data_vars=dict(
                fish_ic=(['nfish', 'X'], fish_ic * np.ones((8, nX))),
                bent_ic=(['nb', 'X'], benthic_prey_ic * np.ones((1, nX))),
            ),
            coords=dict(X=X),
        )
    else:
        ds_ic = xr.open_dataset(ic_file).rename(ic_rename).assign_coords({'X': X})

    if num_chunks > 1:
        chunks = gen_chunks_dict(ds_ic, num_chunks)
        ds_ic = ds_ic.chunk(chunks)

    return ds_ic


def generate_1D_to_2D_pop_map(forcing_file):
    if type(forcing_file) == list:
        ds = xr.open_dataset(forcing_file[0])
    else:
        ds = xr.open_dataset(forcing_file)
    nlat = len(ds['nlat'])
    nlon = len(ds['nlon'])
    ds = ds.stack(X=('nlat', 'nlon'))
    da = xr.zeros_like(ds['HT'])
    da.data = range(len(da.X))
    ds_out = da.where(ds['HT'] > 0, drop=True).astype(np.int64).to_dataset(name='map')
    ds_out['nlat_dimsize'] = nlat
    ds_out['nlon_dimsize'] = nlon
    return ds_out


def map_ds_back_to_2D_pop(ds, map):
    time = len(ds['time'])
    group = len(ds['group'])
    full_data = np.full((time, group, map['nlat_dimsize'].data * map['nlon_dimsize'].data), np.nan)
    full_data[:, :, map['map'].data] = ds['biomass'].data[:, :, :]
    full_data = full_data.reshape(time, group, map['nlat_dimsize'].data, map['nlon_dimsize'].data)
    ds_new = xr.Dataset(
        data_vars=dict(
            biomass=(['time', 'group', 'nlat', 'nlon'], full_data),
        ),
        coords=dict(
            time=ds['time'].data,
            group=ds['group'].data,
        ),
    )
    return ds_new


def gen_chunks_dict(ds, num_chunks):
    return dict(X=(len(ds.X) - 1) // num_chunks + 1)


def generate_single_ds_for_feisty(
    num_chunks,
    forcing_file,
    forcing_rename={},
    allow_negative_forcing=False,
):
    print(f'Starting forcing dataset generation at {time.strftime("%H:%M:%S")}')

    # Read forcing file, and add zooplankton dimension if necessary
    if type(forcing_file) == list:
        ds = xr.open_mfdataset(
            forcing_file, parallel=False, data_vars='minimal', compat='override', coords='minimal'
        ).rename(forcing_rename)

        keep_vars = [
            'bathymetry',
            'poc_flux_bottom',
            'zooC',
            'zoo_mort',
            'T_bottom',
            'T_pelagic',
        ]
        ds = ds[keep_vars]

        # TODO: remove POP specific
        ds = ds.drop_vars(['TLAT', 'TLONG', 'ULAT', 'ULONG'])

    else:
        ds = xr.open_dataset(forcing_file).rename(forcing_rename)

    if 'X' not in ds.dims:
        ds_tmp = ds.stack(X=('nlat', 'nlon'))
        ds = ds_tmp.where(ds_tmp.bathymetry > 0, drop=True)

    if num_chunks > 1:
        chunks = gen_chunks_dict(ds, num_chunks)
        ds = ds.chunk(chunks)

    if not allow_negative_forcing:
        for varname in ['poc_flux_bottom', 'zooC', 'zoo_mort']:
            ds[varname].data = np.where(ds[varname].data > 0, ds[varname].data, 0)

    if 'zooplankton' not in ds.dims:
        ds['zooC'] = ds['zooC'].expand_dims('zooplankton')
        ds['zoo_mort'] = ds['zoo_mort'].expand_dims('zooplankton')
        ds['zooplankton'] = xr.DataArray(['Zoo'], dims='zooplankton')

    return ds


# TODO: this should be agnostic wrt FEISTY groups
def generate_template(
    ds,
    nsteps,
    start_date='0001-01-01',
    groups=np.array(['Sf', 'Sp', 'Sd', 'Mf', 'Mp', 'Md', 'Lp', 'Ld', 'benthic_prey'], dtype='<U12'),
    feeding_link=np.array(
        [
            'Sf_Zoo',
            'Sp_Zoo',
            'Sd_Zoo',
            'Mf_Zoo',
            'Mf_Sf',
            'Mf_Sp',
            'Mf_Sd',
            'Mp_Zoo',
            'Mp_Sf',
            'Mp_Sp',
            'Mp_Sd',
            'Md_benthic_prey',
            'Lp_Mf',
            'Lp_Mp',
            'Lp_Md',
            'Ld_Mf',
            'Ld_Mp',
            'Ld_Md',
            'Ld_benthic_prey',
        ],
        dtype='<U15',
    ),
    fish=np.array(['Sf', 'Sp', 'Sd', 'Mf', 'Mp', 'Md', 'Lp', 'Ld'], dtype='<U2'),
    predator=np.array(
        [
            'Sf',
            'Sp',
            'Sd',
            'Mf',
            'Mf',
            'Mf',
            'Mf',
            'Mp',
            'Mp',
            'Mp',
            'Mp',
            'Md',
            'Lp',
            'Lp',
            'Lp',
            'Ld',
            'Ld',
            'Ld',
            'Ld',
        ],
        dtype='<U2',
    ),
    prey=np.array(
        [
            'Zoo',
            'Zoo',
            'Zoo',
            'Zoo',
            'Sf',
            'Sp',
            'Sd',
            'Zoo',
            'Sf',
            'Sp',
            'Sd',
            'benthic_prey',
            'Mf',
            'Mp',
            'Md',
            'Mf',
            'Mp',
            'Md',
            'benthic_prey',
        ],
        dtype='<U12',
    ),
    diagnostic_names=[],
):
    print(f'Starting template generation at {time.strftime("%H:%M:%S")}')
    # Set chunks
    if 'X' in ds.chunks:
        chunks = {'X': ds.chunks['X']}
    else:
        chunks = {}

    # Create template for output (used for xr.map_blocks)
    # TODO: support alternative calendars
    model_time = xr.cftime_range(
        start=start_date,
        periods=nsteps,
        calendar='noleap',
    )
    X = ds['X'].data

    data_vars_dict = dict(
        biomass=(['time', 'group', 'X'], np.zeros((len(model_time), len(groups), len(X))))
    )
    coords_dict = dict(time=(['time'], model_time), group=(['group'], groups), X=(['X'], X))

    # TODO: better way to add diagnostics
    for diag in diagnostic_names:
        if diag in ['encounter_rate_link', 'consumption_rate_link']:
            data_vars_dict[diag] = (
                ['time', 'feeding_link', 'X'],
                np.zeros((len(model_time), len(feeding_link), len(X))),
            )
            coords_dict['feeding_link'] = ('feeding_link', feeding_link)
            coords_dict['predator'] = ('feeding_link', predator)
            coords_dict['prey'] = ('feeding_link', prey)
        else:
            data_vars_dict[diag] = (
                ['time', 'fish', 'X'],
                np.zeros((len(model_time), len(fish), len(X))),
            )
            coords_dict['fish'] = ('fish', fish)

    ds_out = xr.Dataset(
        data_vars=data_vars_dict,
        coords=coords_dict,
    ).assign_coords({'X': ds['X']})

    if chunks:
        ds_out = ds_out.chunk(chunks)

    return ds_out
