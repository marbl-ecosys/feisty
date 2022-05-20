import time

import numpy as np
import xarray as xr


def generate_ic_ds_for_feisty(
    X, ic_file, chunks={}, fish_ic=1e-5, benthic_prey_ic=2e-3, ic_rename={}
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
    if chunks:
        ds_ic = ds_ic.chunk(chunks)
    return ds_ic


def generate_single_ds_for_feisty(
    num_chunks,
    forcing_file,
    ic_file,
    fish_ic=1e-5,
    benthic_prey_ic=2e-3,
    forcing_rename={},
    ic_rename={},
    allow_negative_forcing=False,
    nX=85813,
):
    print(f'Starting forcing dataset generation at {time.strftime("%H:%M:%S")}')
    if num_chunks > 1:
        nX = 85813
        chunks = dict(X=(nX - 1) // num_chunks + 1)
    else:
        chunks = {}

    # Read forcing file, and add zooplankton dimension if necessary
    ds = xr.open_dataset(forcing_file).rename(forcing_rename)
    if chunks:
        ds = ds.chunk(chunks)
    if not allow_negative_forcing:
        for varname in ['poc_flux_bottom', 'zooC', 'zoo_mort']:
            ds[varname].data = np.where(ds[varname].data > 0, ds[varname].data, 0)
    if 'zooplankton' not in ds.dims:
        ds['zooC'] = ds['zooC'].expand_dims('zooplankton')
        ds['zoo_mort'] = ds['zoo_mort'].expand_dims('zooplankton')
        ds['zooplankton'] = xr.DataArray(['Zoo'], dims='zooplankton')

    # Read initial condition file, add fish_ic and bent_ic to ds
    ds_ic = generate_ic_ds_for_feisty(
        ds.X.data, ic_file, chunks, fish_ic, benthic_prey_ic, ic_rename
    )
    for var in ['fish_ic', 'bent_ic']:
        ds[var] = ds_ic[var]
    return ds


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
