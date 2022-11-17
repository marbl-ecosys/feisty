import time
from dataclasses import dataclass

import numpy as np
import xarray as xr
import yaml


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


def map_da_back_to_2D_pop(da, streams):
    map = _generate_1D_to_2D_pop_map(streams)
    time = len(da['time'])
    group = len(da['group'])
    full_data = np.full((time, group, map['nlat_dimsize'].data * map['nlon_dimsize'].data), np.nan)
    full_data[:, :, map['map'].data] = da.data[:, :, :]
    full_data = full_data.reshape(time, group, map['nlat_dimsize'].data, map['nlon_dimsize'].data)
    ds_new = xr.Dataset(
        data_vars=dict(
            biomass=(['time', 'group', 'nlat', 'nlon'], full_data),
        ),
        coords=dict(
            time=da['time'].data,
            group=da['group'].data,
        ),
    )
    return ds_new['biomass']


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
    start_date='0001-01-01',
    end_date='0001-12-31',
    freq='D',
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
        end=end_date,
        freq=freq,
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


def get_forcing_from_config(feisty_config):
    if 'forcing' not in feisty_config:
        raise KeyError("get_forcing_from_config requires argument with 'forcing' key")
    if 'streams' not in feisty_config['forcing']:
        raise KeyError(
            "get_forcing_from_config requires argument with 'streams' key under 'forcing'"
        )

    feisty_forcing = list()
    if type(feisty_config['forcing']['streams']) == str:
        with open(feisty_config['forcing']['streams']) as f:
            feisty_forcing.append(yaml.safe_load(f))
    else:
        for streams_file in feisty_config['forcing']['streams']:
            with open(streams_file) as f:
                feisty_forcing.append(yaml.safe_load(f))

    return feisty_forcing


def generate_forcing_ds_from_config(feisty_forcing):
    forcing_dses = list()
    for forcing_dict in feisty_forcing:
        forcing_rename = forcing_dict.get('field_rename', {})
        root_dir = forcing_dict.get('root_dir', '.')
        forcing_dses.append(
            xr.open_mfdataset(
                [f'{root_dir}/{filename}' for filename in forcing_dict['files']],
                parallel=False,
                data_vars='minimal',
                compat='override',
                coords='minimal',
            ).rename(forcing_rename)
        )
    forcing_ds = xr.merge(forcing_dses)
    if 'TLAT' in forcing_ds:
        forcing_ds = forcing_ds.drop_vars(['TLAT', 'TLONG', 'ULAT', 'ULONG'])

    # Make sure zooplankton dimension exists
    if 'zooplankton' not in forcing_ds.dims:
        forcing_ds['zooC'] = forcing_ds['zooC'].expand_dims('zooplankton')
        forcing_ds['zoo_mort'] = forcing_ds['zoo_mort'].expand_dims('zooplankton')
        forcing_ds['zooplankton'] = xr.DataArray(['Zoo'], dims='zooplankton')

    if 'X' not in forcing_ds.dims:
        ds_tmp = forcing_ds.stack(X=('nlat', 'nlon'))
        forcing_ds = ds_tmp.where(ds_tmp.bathymetry > 0, drop=True)

    # Make sure all variables are in the dataset
    forcing_vars = [
        'forcing_time',
        'X',
        'zooplankton',
        'bathymetry',
        'T_pelagic',
        'T_bottom',
        'poc_flux_bottom',
        'zooC',
        'zoo_mort',
    ]
    for varname in forcing_vars:
        if varname not in forcing_ds.variables:
            raise KeyError(f'Expecting {varname} in forcing dataset')

    return forcing_ds[forcing_vars]


def _generate_1D_to_2D_pop_map(streams):
    # Read a forcing field to get 2D coordinates
    with open(streams[0]) as f:
        stream_config = yaml.safe_load(f)
    root_dir = stream_config.get('root_dir', '.')
    if type(stream_config['files']) == list:
        forcing_file = stream_config['files'][0]
    else:
        forcing_file = stream_config['files']
    ds = xr.open_dataset(f'{root_dir}/{forcing_file}')

    # Create map needed to recreate 2D output
    nlat = len(ds['nlat'])
    nlon = len(ds['nlon'])
    ds = ds.stack(X=('nlat', 'nlon'))
    da = xr.zeros_like(ds['HT'])
    da.data = range(len(da.X))
    ds_out = da.where(ds['HT'] > 0, drop=True).astype(np.int64).to_dataset(name='map')
    ds_out['nlat_dimsize'] = nlat
    ds_out['nlon_dimsize'] = nlon
    return ds_out
