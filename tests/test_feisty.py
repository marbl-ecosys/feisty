import os

import numpy as np
import xarray as xr

import feisty


def test_1day_testcase():
    testcase = feisty.config_testcase('tanh_shelf', 'cyclic')
    testcase.run(1)
    baseline_file = os.path.join('tests', 'baselines', 'test_case_1day.nc')
    baseline_da = xr.open_dataset(baseline_file)['biomass'].transpose('time', 'group', 'X')
    for coord in ['X', 'time', 'group']:
        baseline_da[coord] = testcase.ds[coord]
    xr.testing.assert_allclose(testcase.ds['biomass'], baseline_da)


def test_1day_locs3():
    kwargs = {'forcing_yaml': 'tests/test_forcing.yaml', 'forcing_key': 'test_locs3'}
    testcase = feisty.config_from_netcdf(
        ignore_year_in_forcing=True, domain_kwargs=kwargs, forcing_kwargs=kwargs
    )
    testcase.run(1)
    baseline_file = os.path.join('tests', 'baselines', 'test_locs3_1day.nc')
    baseline_da = xr.open_dataset(baseline_file)['biomass'].transpose('time', 'group', 'X')
    for coord in ['X', 'time', 'group']:
        baseline_da[coord] = testcase.ds[coord]
    xr.testing.assert_allclose(testcase.ds['biomass'], baseline_da)
