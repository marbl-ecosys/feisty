import numpy as np
import pytest

import feisty

from . import conftest

settings_dict_def = feisty.settings.get_defaults()
model_settings = settings_dict_def['model_settings']
food_web_settings = settings_dict_def['food_web']
zoo_settings = settings_dict_def['zooplankton']
fish_settings = settings_dict_def['fish']
benthic_prey_settings = settings_dict_def['benthic_prey']
reproduction_routing = settings_dict_def['reproduction_routing']

for i in range(len(settings_dict_def['food_web'])):
    settings_dict_def['food_web'][i]['encounter_parameters']['preference'] = np.random.rand()


fish_ic_data = 1e-5
benthic_prey_ic_data = 1e-4

n_zoo = len(settings_dict_def['zooplankton'])
n_fish = len(settings_dict_def['fish'])
n_benthic_prey = 1

NX = 10
NX_2 = 5
domain_dict = {
    'NX': NX,
    'depth_of_seafloor': np.concatenate((np.ones(NX_2) * 1500.0, np.ones(NX_2) * 15.0)),
}

F = feisty.feisty_instance_type(
    settings_dict=settings_dict_def,
    domain_dict=domain_dict,
    fish_ic_data=fish_ic_data,
    benthic_prey_ic_data=benthic_prey_ic_data,
)


def test_set_zoo_biomass():
    """set_zoo_biomass should set the zooplankton biomass, but nothing else"""
    data = np.random.rand(n_zoo, NX)

    zoo_data_prior = F.biomass.isel(group=F.ndx_zoo).data
    fish_data_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data

    F.set_zoo_biomass(data)

    # make sure it worked
    assert (F.biomass.isel(group=F.ndx_zoo).data == data).all()

    # make sure there's no bleed over
    assert (F.biomass.isel(group=F.ndx_fish).data == fish_data_prior).all()
    assert (F.biomass.isel(group=F.ndx_benthic_prey).data == benthic_data_prior).all()

    # put it back
    F.set_zoo_biomass(zoo_data_prior)


def test_set_zoo_mortality():
    """set_zoo_mortality should set the zooplankton mortality"""

    data = np.random.rand(n_zoo, NX)
    F.set_zoo_mortality(data)

    # make sure it worked
    assert (F.zoo_mortality.data == data).all()

    # put it back
    F.set_zoo_mortality(data * 0.0)


def test_set_zoo_mortality_wrong_shape():
    """set_zoo_mortality should raise assertion error if shape is wrong"""
    data = np.random.rand(42, NX * 2)
    with pytest.raises(AssertionError):
        F.set_zoo_mortality(data)


def test_set_fish_biomass():
    """set_fish_biomass should set the fish biomass, but nothing else"""
    data = np.random.rand(n_fish, NX)

    zoo_data_prior = F.biomass.isel(group=F.ndx_zoo).data
    fish_data_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data

    F.set_fish_biomass(data)

    # make sure it worked
    assert (F.biomass.isel(group=F.ndx_fish).data == data).all()

    # make sure there's no bleed over
    assert (F.biomass.isel(group=F.ndx_zoo).data == zoo_data_prior).all()
    assert (F.biomass.isel(group=F.ndx_benthic_prey).data == benthic_data_prior).all()

    # put it back
    F.set_fish_biomass(fish_data_prior)


def test_set_benthic_biomass():
    """set_fish_biomass should set the fish biomass, but nothing else"""
    data = np.random.rand(n_benthic_prey, NX)

    zoo_data_prior = F.biomass.isel(group=F.ndx_zoo).data
    fish_data_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data

    F.set_benthic_prey_biomass(data)

    # make sure it worked
    assert (F.biomass.isel(group=F.ndx_benthic_prey).data == data).all()

    # make sure there's no bleed over
    assert (F.biomass.isel(group=F.ndx_zoo).data == zoo_data_prior).all()
    assert (F.biomass.isel(group=F.ndx_fish).data == fish_data_prior).all()

    # put it back
    F.set_benthic_prey_biomass(benthic_data_prior)


def test_set_biomass_wrong_shape():
    """set_{X}_biomass should raise assertion error if shape is wrong"""
    data = np.random.rand(42, NX * 2)
    with pytest.raises(AssertionError):
        F.set_zoo_biomass(data)
    with pytest.raises(AssertionError):
        F.set_fish_biomass(data)
    with pytest.raises(AssertionError):
        F.set_benthic_prey_biomass(data)


def test_gcm_state_update():
    F.gcm_state.update(
        T_pelagic=25.0,
        T_bottom=2.0,
        poc_flux=33.0,
    )
    assert (F.gcm_state.T_pelagic == 25.0).all()
    assert (F.gcm_state.T_bottom == 2.0).all()
    assert (F.gcm_state.poc_flux == 33.0).all()
