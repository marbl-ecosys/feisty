import os

import yaml

import feisty.driver


def test_idealized_forcing():
    feisty.driver.idealized_forcing(10, 10)


def test_get_gcm_forcing_t():
    pass


def test_read_settings():
    """ensure we can update default settings from a file or dict"""
    sd_default = feisty.settings.get_defaults()
    sd = feisty.driver._read_settings(None)
    assert sd == sd_default

    updates = {'loffline': False}
    sd_updated_dict = feisty.driver._read_settings(updates)
    assert sd_updated_dict != sd_default
    for key, value in updates.items():
        assert sd_updated_dict[key] == value

    file_in = 'test_settings_file.yml'
    with open(file_in, 'w') as fid:
        yaml.dump(updates, fid)

    sd_updated_file = feisty.driver._read_settings(file_in)
    assert sd_updated_file == sd_updated_dict

    os.remove(file_in)
