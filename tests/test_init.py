import numpy as np
import pytest
import xarray as xr

import feisty
import feisty.core.domain as domain
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
    settings_dict_def['food_web'][i]['encounter_parameters']['preference'] = np.random.rand()


fish_ic_data = 1e-5
benthic_prey_ic_data = 1e-4

n_zoo = len(settings_dict_def['zooplankton'])
n_fish = len(settings_dict_def['fish']['members'])
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


def test_reproduction_routing():
    obj = F.reproduction_routing
    assert isinstance(obj, ecosystem.reproduction_routing)
    for i, link in enumerate(obj):
        link_i_expected = reproduction_routing[i]
        if 'is_larval' in link_i_expected:
            assert link.is_larval == link_i_expected['is_larval']
        else:
            assert not link.is_larval
        if 'efficiency' in link_i_expected:
            assert link.efficiency == link_i_expected['efficiency']
        elif not link.is_larval:
            assert link.efficiency is None

        assert F.biomass.group.isel(group=link.ndx_from) == link_i_expected['from']
        assert F.fish[link.i_fish].name == link_i_expected['to']


def test_reproduction_routing_bad_is_larval():
    """ensure that you can't initialize with wrong domain lengths"""
    sd_bad = feisty.settings.get_defaults()
    for i in range(len(sd_bad['reproduction_routing'])):
        if 'is_larval' in sd_bad['reproduction_routing'][i]:
            del sd_bad['reproduction_routing'][i]['efficiency']

    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
        )


def test_domain_values():
    """test domain module init"""

    assert domain._N_points == domain_dict['NX']
    assert (domain.ocean_depth == domain_dict['depth_of_seafloor']).all()


def test_domain_bad_depth_of_seafloor():
    """ensure that you can't initialize with wrong domain lengths"""
    domain_dict_bad = {'NX': NX, 'depth_of_seafloor': np.ones(NX * 2) * 1500.0}
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict_bad,
            fish_ic_data=fish_ic_data,
        )


def test_ecosystem_init():
    """ensure all ecosystem module vars are initialized"""

    # make sure module variables have been initialized
    assert ecosystem._size_class_masses
    assert ecosystem._size_class_bnds_ratio
    assert ecosystem.functional_types
    assert ecosystem.mortality_types

    # check some 1:1 values
    assert (
        ecosystem.PI_be_cutoff
        == settings_dict_def['model_settings']['benthic_pelagic_depth_cutoff']
    )
    assert set(ecosystem._pdc_type_keys) == set(
        settings_dict_def['model_settings']['pelagic_demersal_coupling_type_keys']
    )
    assert ecosystem._pelagic_functional_type_keys == set(
        settings_dict_def['model_settings']['pelagic_functional_type_keys']
    )
    assert ecosystem._demersal_functional_type_keys == set(
        settings_dict_def['model_settings']['demersal_functional_type_keys']
    )


def test_ecosystem_size_class_bounds():
    """ensure size_class_bounds are as expected"""
    size_class_bounds = settings_dict_def['model_settings']['size_class_bounds']
    # ensure size classes init
    for name, size_bounds in size_class_bounds.items():
        assert ecosystem._size_class_masses[name] == np.power(10.0, np.log10(size_bounds).mean())
        assert ecosystem._size_class_bnds_ratio[name] == size_bounds[0] / size_bounds[1]


def test_func_type_init():
    func_types_expected = settings_dict_def['model_settings']['functional_type_keys']

    # ensure all are present
    assert set(ecosystem.functional_types.keys()) == set(func_types_expected)

    # ensure no extras
    assert len(ecosystem.functional_types.keys()) == len(func_types_expected)

    # ensure unique entries
    assert len(set(ecosystem.functional_types.values())) == len(ecosystem.functional_types.values())


def test_bad_func_type_fails():
    """init should fail if there is an unknown type"""
    sd_bad = feisty.settings.get_defaults()
    sd_bad['fish']['members'][0]['functional_type'] = 'UnkownType'
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_bad_mortality_type_fails():
    """init should fail if there is an unknown mortality type"""
    sd_mort = feisty.settings.get_defaults()
    for i in range(len(sd_mort['fish']['members'])):
        sd_mort['fish']['members'][i]['mortality_type'] = 'death by hanging'
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_mort,
        )


def test_bad_pelagic_demersal_coupling_types_fails():
    """init should fail if there is an unknown pelagic_demersal_coupling_type_keys"""
    sd_bad = feisty.settings.get_defaults()
    sd_bad['model_settings']['pelagic_demersal_coupling_type_keys'].append('UnkownType')
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_bad_pelagic_functional_types_fails():
    """init should fail if there is an unknown pelagic_functional_type_keys"""
    sd_bad = feisty.settings.get_defaults()
    sd_bad['model_settings']['pelagic_functional_type_keys'].append('UnkownType')
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_bad_functional_types_apply_pref():
    """init should fail if type is in apply_preference but not in pelagic_demersal_coupling_types"""
    sd_bad = feisty.settings.get_defaults()

    sd_bad['model_settings']['pelagic_demersal_coupling_type_keys'] = [
        'demersal',
        'piscivore',
    ]

    sd_bad['model_settings']['pelagic_demersal_coupling_apply_pref_type_keys'] = [
        'forage',
        'demersal',
    ]
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_bad_demersal_functional_types_fails():
    """init should fail if there is an unknown demersal_functional_type_keys"""
    sd_bad = feisty.settings.get_defaults()
    sd_bad['model_settings']['demersal_functional_type_keys'].append('UnkownType')
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_bad_size_class_fail():
    """init should fail if there is an unknown size class"""
    sd_bad = feisty.settings.get_defaults()
    sd_bad['fish']['members'][0]['size_class'] = 'Eeeeenormous!'
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_duplicated_pelagic_demersal_type_keys():
    sd_bad = feisty.settings.get_defaults()
    combined_list = (
        sd_bad['model_settings']['pelagic_functional_type_keys']
        + sd_bad['model_settings']['demersal_functional_type_keys']
    )
    sd_bad['model_settings']['pelagic_functional_type_keys'] = combined_list
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_zoo_init():
    """zooplankton init should set names and n_zoo"""
    assert F.zoo_names == [z['name'] for z in settings_dict_def['zooplankton']]
    assert F.n_zoo == len(settings_dict_def['zooplankton'])
    assert settings_dict_def['loffline']
    assert F.zoo_mortality.dims == ('zooplankton', 'X')
    assert F.zoo_mortality.shape == (n_zoo, NX)
    assert (F.zoo_mortality.data == 0.0).all()


def test_zoo_mortality_not_offline():
    settings_dict_def_loffline_false = feisty.settings.get_defaults()
    settings_dict_def_loffline_false['loffline'] = False
    Fprime = feisty.feisty_instance_type(
        domain_dict=domain_dict,
        settings_dict=settings_dict_def_loffline_false,
    )
    assert Fprime.zoo_mortality is None


def test_biomass_init():
    """test biomass fields"""

    assert isinstance(F.biomass, xr.DataArray)
    assert F.biomass.dims == ('group', 'X')
    assert F.biomass.shape == (n_zoo + n_fish + 1, NX)
    assert (
        F.biomass.group.isel(group=F.ndx_zoo)
        == [z['name'] for z in settings_dict_def['zooplankton']]
    ).all()
    assert (
        F.biomass.group.isel(group=F.ndx_fish)
        == [f['name'] for f in settings_dict_def['fish']['members']]
    ).all()
    assert (
        F.biomass.group.isel(group=F.ndx_benthic_prey)
        == [b['name'] for b in settings_dict_def['benthic_prey']]
    ).all()


def test_biomass_init_values():
    """biomass initialization should reflect inputs"""
    assert (F.biomass.isel(group=F.ndx_fish).data == fish_ic_data).all()
    assert (F.biomass.isel(group=F.ndx_zoo).data == 0.0).all()
    assert (F.biomass.isel(group=F.ndx_benthic_prey).data == benthic_prey_ic_data).all()


def test_biomass_bad_shape_benthic_prey():
    # ensure that initializing with data that doesn't conform fails
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            benthic_prey_ic_data=np.arange(25),
        )


def test_biomass_bad_shape_fish():
    # ensure that initializing with data that doesn't conform fails
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            fish_ic_data=np.arange(25),
        )


def test_duplicate_names_across_groups():
    sd_bad = feisty.settings.get_defaults()
    sd_bad['zooplankton'][0]['name'] = sd_bad['fish']['members'][0]['name']
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_gcm_state():
    # ensure gcm_state conforms
    assert isinstance(F.gcm_state, feisty.core.interface.gcm_state_type)
    assert isinstance(F.gcm_state.T_pelagic, xr.DataArray)
    assert isinstance(F.gcm_state.T_bottom, xr.DataArray)
    assert isinstance(F.gcm_state.poc_flux, xr.DataArray)


def test_fishing():
    assert isinstance(F.fishing, ecosystem.fishing)
    assert (
        F.fishing.fishing_rate == settings_dict_def['fishing']['fishing_rate_per_year'] / 365.0
    ).all()


def test_init_tendency_arrays():
    """test init tendency"""

    ds = F.tendency_data
    assert isinstance(ds, xr.Dataset)
    assert (ds.zooplankton == [f['name'] for f in zoo_settings]).all()
    assert (ds.fish == [f['name'] for f in fish_settings['members']]).all()
    assert (ds.benthic_prey == [b['name'] for b in benthic_prey_settings]).all()

    assert set(ds.coords.keys()) == {'zooplankton', 'fish', 'benthic_prey'}

    checked = []
    for key, da in ds.data_vars.items():
        if key in [
            't_frac_pelagic',
            'T_habitat',
            'ingestion_rate',
            'predation_flux',
            'predation_rate',
            'metabolism_rate',
            'mortality_rate',
            'energy_avail_rate',
            'growth_rate',
            'reproduction_rate',
            'recruitment_flux',
            'total_tendency',
            'fish_catch_rate',
        ]:
            assert da.dims == ('fish', 'X')
            assert da.shape == (n_fish, NX)
            checked.append(key)
        elif key in ['benthic_biomass_new']:
            assert da.dims == ('benthic_prey', 'X')
            assert da.shape == (n_benthic_prey, NX)
            checked.append(key)

    assert set(checked) == set(ds.data_vars)
