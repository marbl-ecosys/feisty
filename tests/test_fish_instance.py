import numpy as np
import pytest
import xarray as xr

import feisty
import feisty.core.ecosystem as ecosystem
import feisty.core.settings as settings

from . import conftest

settings_dict_def = settings.get_defaults()
model_settings = settings_dict_def['model_settings']
food_web_settings = settings_dict_def['food_web']
zoo_settings = settings_dict_def['zooplankton']
fish_settings = settings_dict_def['fish']
benthic_prey_settings = settings_dict_def['benthic_prey']
reproduction_routing = settings_dict_def['reproduction_routing']

for i in range(len(settings_dict_def['food_web'])):
    settings_dict_def['food_web'][i]['preference'] = np.random.rand()


fish_ic_data = 1e-5
benthic_prey_ic_data = 1e-4

n_zoo = len(settings_dict_def['zooplankton'])
n_fish = len(settings_dict_def['fish']['members'])
n_benthic_prey = 1

NX = 10
NX_2 = 5
domain_dict = {
    'NX': NX,
    'bathymetry': np.concatenate((np.ones(NX_2) * 1500.0, np.ones(NX_2) * 15.0)),
}

F = feisty.feisty_instance_type(
    settings_dict=settings_dict_def,
    domain_dict=domain_dict,
    fish_ic_data=fish_ic_data,
    benthic_prey_ic_data=benthic_prey_ic_data,
)


def test_fish_init():
    """fish init should yield consistent information"""

    assert F.n_fish == n_fish
    for i, fish in enumerate(F.fish):
        assert fish.name == fish_settings['members'][i]['name']
        assert fish.size_class == fish_settings['members'][i]['size_class']
        assert fish.functional_type_key == fish_settings['members'][i]['functional_type']
        assert fish.functional_type == ecosystem.functional_types[fish.functional_type_key]
        assert fish.t_frac_pelagic_static == fish_settings['members'][i]['t_frac_pelagic_static']
        assert (
            fish.pelagic_demersal_coupling
            == fish_settings['members'][i]['pelagic_demersal_coupling']
        )
        assert fish.mass == ecosystem._size_class_masses[fish.size_class]

        assert isinstance(fish.t_frac_pelagic, xr.DataArray)
        assert fish.t_frac_pelagic.name == f'{fish.name}_t_frac_pelagic'
        assert (fish.t_frac_pelagic == fish_settings['members'][i]['t_frac_pelagic_static']).all()
        assert fish.harvest_selectivity == fish_settings['members'][i]['harvest_selectivity']
        assert (
            fish.energy_frac_somatic_growth
            == fish_settings['members'][i]['energy_frac_somatic_growth']
        )
        assert (
            fish.__repr__()
            == f"{fish_settings['members'][i]['name']}: {fish_settings['members'][i]['size_class']} {fish_settings['members'][i]['functional_type']}"
        )


def test_fish_init_duplicate_fish():
    """ensure that we cannot initialize with duplicate fish"""

    settings_dict_def_duplicate_fish = settings.get_defaults()
    settings_dict_def_duplicate_fish['fish']['members'].append(
        settings_dict_def_duplicate_fish['fish']['members'][0]
    )
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=settings_dict_def_duplicate_fish,
        )


def test_fish_defaults():
    """ensure that we cannot initialize bad harvest_selectivity"""

    settings_dict_def_bad = settings.get_defaults()

    fish0_parms = settings_dict_def_bad['fish']['members'][0]
    bad_kwargs = dict(**settings_dict_def_bad['fish']['defaults'])
    bad_kwargs['stink_factor'] = 86.5

    # this should work:
    group_ind = 0
    fish = feisty.core.ecosystem.fish_type(group_ind, **fish0_parms)
    for key, value in settings_dict_def_bad['fish']['defaults'].items():
        assign_key = key
        assign_value = value
        if key == 'mortality_coeff_per_yr':
            assign_key = 'mortality_coeff'
            assign_value = assign_value / 365.0
        elif key == 'mortality_type':
            assign_value = feisty.core.ecosystem.mortality_types[assign_value]
        assert fish.__dict__[assign_key] == assign_value

    # this should fail
    fish0_parms.update(bad_kwargs)
    with pytest.raises(ValueError):
        group_ind = 0
        feisty.core.ecosystem.fish_type(group_ind, **fish0_parms)


def test_fish_bad_harvest_selectivity():
    """ensure that we cannot initialize bad harvest_selectivity"""

    settings_dict_def_duplicate_fish = settings.get_defaults()
    settings_dict_def_duplicate_fish['fish']['members'][0]['harvest_selectivity'] = 99.0
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=settings_dict_def_duplicate_fish,
        )


def test_fish_bad_energy_frac_somatic_growth():
    """ensure that we cannot initialize bad energy_frac_somatic_growth"""

    settings_dict_def_duplicate_fish = settings.get_defaults()
    settings_dict_def_duplicate_fish['fish']['members'][0]['energy_frac_somatic_growth'] = 99.0
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=settings_dict_def_duplicate_fish,
        )


def test_fish_init_uncoupled():
    """ensure that we cannot initialize fish with pelagic_demersal_coupling = True if functional_type is not in pelagic_demersal_coupling_type_keys"""
    settings_dict_def_fish_bad_pdc = settings.get_defaults()

    for i in range(len(settings_dict_def_fish_bad_pdc['fish']['members'])):
        if settings_dict_def_fish_bad_pdc['fish']['members'][i]['functional_type'] == 'forage':
            settings_dict_def_fish_bad_pdc['fish']['members'][i]['pelagic_demersal_coupling'] = True

    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=settings_dict_def_fish_bad_pdc,
        )


def test_fish_apply_pref():
    pdc_apply_pref_type_keys = settings_dict_def['model_settings'][
        'pelagic_demersal_coupling_apply_pref_type_keys'
    ]
    for i, fish in enumerate(F.fish):
        pdc_apply_pref = fish_settings['members'][i]['functional_type'] in pdc_apply_pref_type_keys
        assert fish.pdc_apply_pref == pdc_apply_pref


def test_is_demersal():
    for obj in F.member_obj_list:
        is_demersal = obj.functional_type_key in model_settings['demersal_functional_type_keys']
        assert is_demersal == obj.is_demersal

        is_zooplankton = (
            obj.functional_type_key in model_settings['zooplankton_functional_type_keys']
        )
        assert is_zooplankton == obj.is_zooplankton
