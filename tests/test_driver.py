import os

import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

import feisty


def test_gen_domain():
    domain_dict = feisty.testcase.idealized.gen_domain_dict(np.arange(0, 10), np.ones(10) * 50.0)
    assert domain_dict['NX'] == 10
    assert (domain_dict['bathymetry'] == 50.0).all()


def test_forcing_cyclic():
    forcing = feisty.testcase.forcing_cyclic(feisty.testcase.domain_tanh_shelf())
    assert set(forcing.data_vars) == {
        'T_pelagic',
        'T_bottom',
        'poc_flux_bottom',
        'zooC',
        'zoo_mort',
    }


def test_not_implemented():
    """ensure appropriate failures with bad method."""
    with pytest.raises(ValueError):
        feisty.config_and_run_testcase('tanh_shelf', 'cyclic', method='intuition')
    with pytest.raises(NotImplementedError):
        feisty.config_and_run_testcase('tanh_shelf', 'cyclic', method='Radau')


def test_read_settings():
    """ensure we can update default settings from a file or dict"""
    sd_default = feisty.settings.get_defaults()
    sd = feisty.offline_driver_mod._read_settings(None)
    assert sd == sd_default

    updates = {'loffline': False}
    sd_updated_dict = feisty.offline_driver_mod._read_settings(updates)
    assert sd_updated_dict != sd_default
    for key, value in updates.items():
        assert sd_updated_dict[key] == value

    file_in = 'test_settings_file.yml'
    with open(file_in, 'w') as fid:
        yaml.dump(updates, fid)

    sd_updated_file = feisty.offline_driver_mod._read_settings(file_in)
    assert sd_updated_file == sd_updated_dict

    os.remove(file_in)


# def test_config_testcase_init_1():
#     testcase = feisty.config_testcase('tanh_shelf', 'cyclic')
#     assert isinstance(testcase, feisty.offline_driver)
#     for attr in ['obj', 'domain_dict', 'forcing', 'settings_in', 'run']:
#         assert hasattr(testcase, attr)

#     assert isinstance(testcase.forcing, xr.Dataset)
#     assert isinstance(testcase.obj, feisty.feisty_instance_type)
#     assert testcase.domain_dict['NX'] == feisty.testcase.domain_tanh_shelf()['NX']
#     assert (
#         testcase.domain_dict['bathymetry'] == feisty.testcase.domain_tanh_shelf()['bathymetry']
#     ).all()

#     expected_forcing = feisty.testcase.forcing_cyclic(feisty.testcase.domain_tanh_shelf())
#     for v in testcase.forcing.data_vars:
#         assert (testcase.forcing[v] == expected_forcing[v]).all()


# def test_config_testcase_init_2():
#     testcase = feisty.config_testcase('tanh_shelf', 'cyclic')
#     testcase.state_t = (
#         testcase.obj.get_prognostic().copy().assign_coords({'X': testcase.forcing.X.data})
#     )

#     testcase._init_output_arrays(365)
#     assert (
#         testcase._ds_list[0].time
#         == xr.cftime_range(start=cftime.DatetimeNoLeap(1, 1, 1), periods=365)
#     ).all()
#     assert isinstance(testcase._ds_list[0], xr.Dataset)
#     assert set(testcase._ds_list[0].data_vars) == {'biomass'}.union(testcase._diagnostic_names)
#     assert len(testcase._ds_list[0].group) == len(testcase.obj.ndx_prognostic)


def test_config_testcase_run():
    feisty.config_and_run_testcase('tanh_shelf', 'cyclic', end_date='0001-01-03')


def test_cyclic_interpolation():
    ds1 = feisty.config_and_run_testcase('tanh_shelf', 'cyclic')
    ds2 = feisty.config_and_run_testcase(
        'tanh_shelf', 'cyclic', start_date='0002-01-01', end_date='0002-01-01'
    )
    assert (ds1['biomass'].data == ds2['biomass'].data).all()
