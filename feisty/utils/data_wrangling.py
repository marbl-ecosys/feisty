import datetime
import os
import shutil
import time
from dataclasses import dataclass

import dask
import numpy as np
import xarray as xr
import yaml
import zarr


def generate_ic_ds_for_feisty(
    ds, ic_file=None, fish_ic=1e-5, benthic_prey_ic=2e-3, ic_rename={}, chunks={}
):
    if ic_file is None:
        if 'X' in ds.coords:
            nX = len(ds.X)
            ds_ic = xr.Dataset(
                data_vars=dict(
                    fish_ic=(['nfish', 'X'], fish_ic * _ones((8, nX), chunks)),
                    bent_ic=(['nb', 'X'], benthic_prey_ic * _ones((1, nX), chunks)),
                ),
                coords=dict(X=ds.X.data),
            )
        else:
            nlat = len(ds.nlat)
            nlon = len(ds.nlon)
            ds_ic = xr.Dataset(
                data_vars=dict(
                    fish_ic=(['nfish', 'nlat', 'nlon'], fish_ic * _ones((8, nlat, nlon), chunks)),
                    bent_ic=(
                        ['nb', 'nlat', 'nlon'],
                        benthic_prey_ic * _ones((1, nlat, nlon), chunks),
                    ),
                ),
                coords=dict(nlat=ds.nlat.data, nlon=ds.nlon.data),
            )
    else:
        if ic_file[-3:] == '.nc':
            ds_ic = xr.open_dataset(ic_file).rename(ic_rename)
        elif ic_file[-5:] == '.zarr':
            ds_ic = xr.open_zarr(ic_file).rename(ic_rename)
        else:
            raise ValueError(f'Can not determine file type for {ic_file}')
        if 'X' in ds.coords:
            ds_ic = ds_ic.assign_coords(dict(X=ds.X.data))
        else:
            ds_ic = ds_ic.assign_coords(dict(nlat=ds.nlat.data, nlon=ds.nlon.data))

    # if num_chunks > 1:
    #     chunks = gen_chunks_dict(ds_ic, num_chunks)
    #     ds_ic = ds_ic.chunk(chunks).persist()

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
        ds = ds.stack(X=('nlat', 'nlon'))  # .where(ds_tmp.bathymetry > 0, drop=True)

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
    forcings=np.array(
        [
            'zooC',
            'zoo_mort',
            'T_bottom',
            'T_pelagic',
            'poc_flux',
        ],
        dtype='<U12',
    ),
    diagnostic_names=[],
):
    print(f'Starting template generation at {time.strftime("%H:%M:%S")}')
    # Create template for output (used for xr.map_blocks)
    # TODO: support alternative calendars
    model_time = xr.cftime_range(
        start=start_date,
        end=end_date,
        freq=freq,
        calendar='noleap',
    )

    # Set chunks and dims
    coords_dict = dict(time=(['time'], model_time), group=(['group'], groups))
    biomass_dims = ['time', 'group']
    biomass_dimsizes = [len(model_time), len(groups)]
    feeding_dims = ['time', 'feeding_link']
    feeding_dimsizes = [len(model_time), len(feeding_link)]
    fish_dims = ['time', 'fish']
    fish_dimsizes = [len(model_time), len(fish)]
    forcings_dims = ['time', 'forcings']
    forcings_dimsizes = [len(model_time), len(forcings)]
    if 'X' in ds.coords:
        check_chunk = ['X']
        coords_dict['X'] = (['X'], ds.X.data)
        biomass_dims.append('X')
        biomass_dimsizes.append(len(ds.X))
        feeding_dims.append('X')
        feeding_dimsizes.append(len(ds.X))
        fish_dims.append('X')
        fish_dimsizes.append(len(ds.X))
        forcings_dims.append('X')
        forcings_dimsizes.append(len(ds.X))
    else:
        check_chunk = ['nlat', 'nlon']
        coords_dict['nlat'] = (['nlat'], ds.nlat.data)
        coords_dict['nlon'] = (['nlon'], ds.nlon.data)
        biomass_dims.append('nlat')
        biomass_dimsizes.append(len(ds.nlat))
        biomass_dims.append('nlon')
        biomass_dimsizes.append(len(ds.nlon))
        feeding_dims.append('nlat')
        feeding_dimsizes.append(len(ds.nlat))
        feeding_dims.append('nlon')
        feeding_dimsizes.append(len(ds.nlon))
        fish_dims.append('nlat')
        fish_dimsizes.append(len(ds.nlat))
        fish_dims.append('nlon')
        fish_dimsizes.append(len(ds.nlon))
        forcings_dims.append('nlat')
        forcings_dimsizes.append(len(ds.nlat))
        forcings_dims.append('nlon')
        forcings_dimsizes.append(len(ds.nlon))

    chunks = {}
    for dimname in check_chunk:
        if dimname in ds.chunks:
            chunks[dimname] = ds.chunks[dimname]

    data_vars_dict = dict(biomass=(biomass_dims, _zeros(biomass_dimsizes, chunks)))

    # TODO: better way to add diagnostics
    for diag in diagnostic_names:
        if diag in ['encounter_rate_link', 'consumption_rate_link']:
            data_vars_dict[diag] = (feeding_dims, _zeros(feeding_dimsizes, chunks))
            coords_dict['feeding_link'] = ('feeding_link', feeding_link)
            coords_dict['predator'] = ('feeding_link', predator)
            coords_dict['prey'] = ('feeding_link', prey)
        elif diag == 'forcing_data':
            data_vars_dict[diag] = (forcings_dims, _zeros(forcings_dimsizes, chunks))
            coords_dict['forcings'] = ('forcings', forcings)
        else:
            data_vars_dict[diag] = (fish_dims, _zeros(fish_dimsizes, chunks))
            coords_dict['fish'] = ('fish', fish)

    ds_out = xr.Dataset(
        data_vars=data_vars_dict,
        coords=coords_dict,
    )
    if 'X' in coords_dict:
        ds_out = ds_out.assign_coords({'X': ds.X})

    # if chunks:
    # ds_out = ds_out.chunk(chunks)

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


def generate_forcing_ds_from_config(feisty_forcing, chunks, POP_units=False, debug_outdir=None):
    forcing_dses = list()
    for forcing_dict in feisty_forcing:
        forcing_rename = forcing_dict.get('field_rename', {})
        root_dir = forcing_dict.get('root_dir', '.')
        day_offset = forcing_dict.get('day_offset', 0)

        default_chunks = {
            'forcing_time': -1,  # entire time series as one chunk
            'nlat': 128,
            'nlon': 80,
        }

        new_ds = xr.open_mfdataset(
            [os.path.join(root_dir, filename) for filename in forcing_dict['files']],
            data_vars='minimal',
            compat='override',
            coords='minimal',
            decode_timedelta=True,
        ).rename(forcing_rename)

        new_ds = new_ds.chunk(default_chunks)

        new_ds = new_ds.assign_coords(
            {'forcing_time': new_ds.forcing_time + datetime.timedelta(day_offset)}
        )
        forcing_dses.append(new_ds)

    forcing_ds = xr.merge(forcing_dses, compat='override', join='override')

    drop_vars = [var for var in ['TLAT', 'TLONG', 'ULAT', 'ULONG'] if var in forcing_ds]
    if drop_vars:
        forcing_ds = forcing_ds.drop_vars(drop_vars)

    # Make sure zooplankton dimension exists
    if 'zooplankton' not in forcing_ds.dims:
        forcing_ds['zooC'] = forcing_ds['zooC'].expand_dims('zooplankton')
        forcing_ds['zoo_mort'] = forcing_ds['zoo_mort'].expand_dims('zooplankton')
        forcing_ds['zooplankton'] = xr.DataArray(['Zoo'], dims='zooplankton')

    # if 'X' not in forcing_ds.dims:
    #     forcing_ds = forcing_ds.stack(X=('nlat', 'nlon'))
    #     forcing_ds = forcing_ds.where(forcing_ds.bathymetry > 0, drop=True)

    # Make sure all variables are in the dataset
    forcing_vars = [
        'forcing_time',
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

    for coordname in ['nlat', 'nlon']:
        if coordname in forcing_ds.dims:
            forcing_ds = forcing_ds.assign_coords(
                {coordname: np.arange(forcing_ds.sizes[coordname])}
            )

    if POP_units:
        # Conversion from Colleen:
        # 1e9 nmol in 1 mol C
        # 1e4 cm2 in 1 m2
        # 12.01 g C in 1 mol C
        # 1 g dry W in 9 g wet W (Pauly & Christiansen)
        nmol_cm2_TO_g_m2 = 1e-9 * 1e4 * 12.01 * 9.0
        per_s_TO_per_d = 86400
        # Depth: cm -> m
        forcing_ds['bathymetry'].data = forcing_ds['bathymetry'].data * 0.01

        # poc_flux_bottom: nmol cm-2 s-1 -> g m-2 d-1
        forcing_ds['poc_flux_bottom'].data = (
            forcing_ds['poc_flux_bottom'].data * nmol_cm2_TO_g_m2 * per_s_TO_per_d
        )

        # zooC: mmol m-3 cm (= nmol cm-2) -> g m-2
        forcing_ds['zooC'].data = forcing_ds['zooC'].data * nmol_cm2_TO_g_m2

        # zoo_mort: mmol m-3 cm s-1 (= nmol cm-2 s-1) -> g m-2 d-1
        forcing_ds['zoo_mort'].data = (
            forcing_ds['zoo_mort'].data * nmol_cm2_TO_g_m2 * per_s_TO_per_d
        )

    if debug_outdir is None:
        raise ValueError('You must manually specify debug_outdir to avoid overwriting Zarr files.')
    print(f'Saving forcing dataset to {debug_outdir}')

    if os.path.exists(debug_outdir):
        print(f'Removing existing Zarr directory at {debug_outdir}')
        shutil.rmtree(debug_outdir)

    print(f'Saving forcing dataset to {debug_outdir}')

    forcing_ds = forcing_ds.chunk(default_chunks)

    forcing_ds.to_zarr(debug_outdir, mode='w', consolidated=True)

    zarr.consolidate_metadata(debug_outdir)

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


def _chunk_dict_to_array(dimsizes, chunks):
    new_chunks = list(dimsizes)
    if 'X' in chunks:
        new_chunks[-1] = chunks['X']
    if 'nlon' in chunks:
        new_chunks[-1] = chunks['nlon']
    if 'nlat' in chunks:
        new_chunks[-2] = chunks['nlat']
    return new_chunks


def _zeros(dimsizes, chunks={}):
    """
    if chunked data is requested, return dask.array.zeros
    otherwise return np.zeros
    """
    if chunks:
        return dask.array.zeros(dimsizes, chunks=_chunk_dict_to_array(dimsizes, chunks))
    return np.zeros(dimsizes)


def _ones(dimsizes, chunks={}):
    """
    if chunked data is requested, return dask.array.ones
    otherwise return np.ones
    """
    if chunks:
        return dask.array.ones(dimsizes, chunks=_chunk_dict_to_array(dimsizes, chunks))
    return np.ones(dimsizes)


def write_restart_file(
    ds,
    rest_file,
    fish_names=['Sf', 'Sp', 'Sd', 'Mf', 'Mp', 'Md', 'Lp', 'Ld'],
    benthic_names=['benthic_prey'],
    overwrite=False,
):
    fish_ic = (
        ds.isel(time=-1)
        .sel(group=fish_names)
        .drop(['time', 'group'])
        .biomass.rename({'group': 'nfish'})
        .to_dataset(name='fish_ic')
    )
    bent_ic = (
        ds.isel(time=-1)
        .sel(group=benthic_names)
        .drop(['time', 'group'])
        .biomass.rename({'group': 'nb'})
        .to_dataset(name='bent_ic')
    )
    new_ic = xr.merge([fish_ic, bent_ic])
    _write_to_nc_or_zarr(new_ic, rest_file, overwrite)


def write_history_file(ds, hist_file, overwrite=False):
    _write_to_nc_or_zarr(ds, hist_file, overwrite)


def _write_to_nc_or_zarr(ds, filename, overwrite=False):
    if os.path.isfile(filename):
        if not overwrite:
            raise ValueError(f'{filename} exists; set overwrite=True to replace')
        print(f'Removing {filename} before writing new copy')
        os.remove(filename)
    elif os.path.isdir(filename):
        if not overwrite:
            raise ValueError(f'{filename}/ exists; set overwrite=True to replace')
        print(f'Removing {filename}/ before writing new copy')
        shutil.rmtree(filename)

    print(f'Writing {filename}')
    if filename[-3:] == '.nc':
        for v in ds.variables:
            ds[v].encoding['_FillValue'] = 1e34
        print('Calling to_netcdf...')
        ds.to_netcdf(filename)
    elif filename[-5:] == '.zarr':
        print('Calling to_zarr...')
        # highres history file needs to be written variable by variable
        # otherwise to_zarr() hangs
        print(f'data_vars! {ds.data_vars}')
        if 'biomass' in ds.data_vars or 'forcings' in ds.data_vars:
            if 'biomass' in ds.data_vars:
                print('Writing biomass')
                ds['biomass'].to_dataset().to_zarr(filename)
            if 'forcings' in ds.data_vars:
                print('Writing forcings')
                ds['forcings'].to_dataset().to_zarr(filename)
            for var in ds.data_vars:
                if var in ['biomass', 'forcings']:
                    continue
                print(f'Writing {var} to disk')
                ds[var].to_dataset().to_zarr(filename, mode='a')
        else:
            ds.to_zarr(filename)
    else:
        raise ValueError(f'Can not determine file type for {filename}')
