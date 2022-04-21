import numpy as np
import xarray as xr


def _comparison(xr_matlab_vals, xr_py_vals, row_name, full_table, seps, thres):
    nstep = np.min([len(xr_matlab_vals['time'].data), len(xr_py_vals['time'].data)])
    nx = np.min([len(xr_matlab_vals['X'].data), len(xr_py_vals['X'].data)])
    matlab_vals = xr_matlab_vals.isel(time=slice(0, nstep), X=slice(0, nx)).data
    py_vals = xr_py_vals.isel(time=slice(0, nstep), X=slice(0, nx)).data
    if thres:
        thres_mask = np.logical_and(np.abs(py_vals) > thres, np.abs(matlab_vals) > thres)
        py_vals = np.where(thres_mask, py_vals, 0)
        matlab_vals = np.where(thres_mask, matlab_vals, 0)
    rel_err_denom = np.where(
        matlab_vals != 0, matlab_vals, 1
    )  # avoid dividing by 0 in next np.where() statement
    if np.any(np.isnan(py_vals)):
        print(f'python has nans for {row_name}!')
        return
    rel_errs = np.where(matlab_vals != 0, np.abs((py_vals - matlab_vals) / rel_err_denom), np.nan)
    rel_errs = np.where(np.logical_and(np.isnan(rel_errs), py_vals == 0), 0, rel_errs)
    rel_errs = np.where(np.logical_and(np.isnan(rel_errs), py_vals > 0), np.inf, rel_errs)
    rel_errs = np.where(np.logical_and(np.isnan(rel_errs), py_vals < 0), -np.inf, rel_errs)
    absval_rel_errs = np.abs(rel_errs)

    if full_table or np.max(absval_rel_errs) > 1e-12:
        max_inds = np.nonzero(np.abs(rel_errs) == np.max(absval_rel_errs))
        t_ind = max_inds[0][0]
        x_ind = max_inds[1][0]
        matlab_val = matlab_vals[t_ind][x_ind]
        py_val = py_vals[t_ind][x_ind]
        rel_err = rel_errs[t_ind][x_ind]
        t_val = xr_py_vals.isel(time=t_ind).time.data
        x_val = xr_py_vals.isel(X=x_ind).X.data
        print(
            f'{seps[0]}{row_name} (t={t_val}, X={x_val}){seps[1]}{matlab_val:10.4e}{seps[2]}'
            + f'{py_val:10.4e}{seps[2]}{rel_err:10.4e}{seps[-1]}'
        )


def compare_nc(baseline_ds, test_da_or_ds, full_table=True, markdown_formatting='true', thres=None):
    if type(test_da_or_ds) == xr.Dataset:
        ds = test_da_or_ds
        da = None
        baseline_da = None
        table_header = ''
    else:
        da = test_da_or_ds
        ds = None
        da_dims = ['group', 'fish']
        da_dim = ''
        for dad in da_dims:
            if dad in da.dims:
                da_dim = dad
                break
        baseline_da = baseline_ds[da.name]
        table_header = da_dim
    if markdown_formatting:
        print(f'| {table_header} | Matlab Value | Python Value | Rel Err |')
        print('| --- | --- | --- | --- |')
        seps = ['| ', ' | ', ' | ', ' |']
    else:
        seps = ['', ': ', ', ', '']
    if da is not None:
        for n, dimname in enumerate(da[da_dim]):
            _comparison(
                baseline_da.isel({da_dim: n}),
                da.isel({da_dim: n}),
                dimname.data,
                full_table,
                seps,
                thres,
            )
    else:
        for varname in ds:
            _comparison(
                baseline_ds.isel(zooplankton=0)[varname],
                ds.isel(zooplankton=0)[varname],
                varname,
                full_table,
                seps,
                thres,
            )


def default_config():
    driver_config = dict()
    # default settings
    for matlab_script in ['test_case', 'test_locs3', 'FOSI_cesm', 'FOSI_cesm_daily', 'FOSI_spinup']:
        driver_config[matlab_script] = dict()

        # Baseline file
        driver_config[matlab_script]['baseline'] = f'{matlab_script}.nc'
        driver_config[matlab_script]['baseline_compare'] = True

        # Generate table to compare forcing to matlab
        driver_config[matlab_script]['compare_forcing'] = True

        # When computing table to compare forcing to matlab, replace negative values of poc_flux_bottom, zooC, and zooC_mort with 0
        driver_config[matlab_script]['force_nonnegative'] = True

        # Model date at beginning of simulation
        driver_config[matlab_script]['start_date'] = '0001-01-01'

        # Length of run (in years)
        driver_config[matlab_script]['nyears'] = 1

        # Size of each entry in testcase._ds
        driver_config[matlab_script]['max_output_time_dim'] = 365

        # Cycle the forcing (assumes a single year of forcing is provided)
        driver_config[matlab_script]['ignore_year_in_forcing'] = False

        # List of diagnostics to include in output
        driver_config[matlab_script]['diagnostic_names'] = []

        # Dictionary used to verride FEISTY settings from feisty/core/default_settings.yml
        driver_config[matlab_script]['settings_in'] = {}

        # Run cells that generate plots?
        driver_config[matlab_script]['make_plots'] = True
        driver_config[matlab_script]['make_err_plots'] = False

        # Default plot settings
        driver_config[matlab_script]['plot_settings'] = dict()
        # Column to plot output from
        driver_config[matlab_script]['plot_settings']['X'] = 0
        # y limits for biomass plots
        driver_config[matlab_script]['plot_settings']['ylim'] = [5e-7, 50]

    return driver_config


def old_settings():
    settings_in = dict()
    settings_in['benthic_prey'] = {
        'defaults': {'benthic_efficiency': 0.075, 'carrying_capacity': 0},
        'members': [{'name': 'benthic_prey'}],
    }
    settings_in['food_web'] = [
        {'predator': 'Sf', 'prey': 'Zoo', 'preference': 1.0},
        {'predator': 'Sp', 'prey': 'Zoo', 'preference': 1.0},
        {'predator': 'Sd', 'prey': 'Zoo', 'preference': 1.0},
        {'predator': 'Mf', 'prey': 'Zoo', 'preference': 0.45},
        {'predator': 'Mf', 'prey': 'Sf', 'preference': 1.0},
        {'predator': 'Mf', 'prey': 'Sp', 'preference': 1.0},
        {'predator': 'Mf', 'prey': 'Sd', 'preference': 1.0},
        {'predator': 'Mp', 'prey': 'Zoo', 'preference': 0.45},
        {'predator': 'Mp', 'prey': 'Sf', 'preference': 1.0},
        {'predator': 'Mp', 'prey': 'Sp', 'preference': 1.0},
        {'predator': 'Mp', 'prey': 'Sd', 'preference': 1.0},
        {'predator': 'Md', 'prey': 'benthic_prey', 'preference': 1.0},
        {'predator': 'Lp', 'prey': 'Mf', 'preference': 0.5},
        {'predator': 'Lp', 'prey': 'Mp', 'preference': 1.0},
        {'predator': 'Lp', 'prey': 'Md', 'preference': 1.0},
        {'predator': 'Ld', 'prey': 'Mf', 'preference': 0.375},
        {'predator': 'Ld', 'prey': 'Mp', 'preference': 0.75},
        {'predator': 'Ld', 'prey': 'Md', 'preference': 1.0},
        {'predator': 'Ld', 'prey': 'benthic_prey', 'preference': 1.0},
    ]
    return settings_in
