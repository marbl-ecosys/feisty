import os

import numpy as np
import xarray as xr

import feisty


def test_1day_testcase():
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
    ds_out = feisty.config_and_run_testcase('tanh_shelf', 'cyclic', 1, settings_in=settings_in)
    baseline_file = os.path.join('tests', 'baselines', 'test_case_1day.nc')
    baseline_da = xr.open_dataset(baseline_file)['biomass'].transpose('time', 'group', 'X')
    for coord in ['X', 'time', 'group']:
        baseline_da[coord] = ds_out[coord]
    xr.testing.assert_allclose(ds_out['biomass'], baseline_da)


def test_1day_locs3():
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
    ds_out = feisty.config_and_run_from_netcdf(
        'tests/test_forcing.yaml',
        'test_locs3',
        1,
        ignore_year_in_forcing=True,
        settings_in=settings_in,
    )
    baseline_file = os.path.join('tests', 'baselines', 'test_locs3_1day.nc')
    baseline_da = xr.open_dataset(baseline_file)['biomass'].transpose('time', 'group', 'X')
    for coord in ['X', 'time', 'group']:
        baseline_da[coord] = ds_out[coord]
    xr.testing.assert_allclose(ds_out['biomass'], baseline_da)
