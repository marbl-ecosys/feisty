import numpy as np
import pandas as pd
import pytest
import xarray as xr

import feisty
import feisty.core.ecosystem as ecosystem

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

all_prey, preference = conftest.get_all_prey_and_preference(settings_dict_def)
fish_func_type = conftest.get_fish_func_type(settings_dict_def)


zoo_names = [z['name'] for z in settings_dict_def['zooplankton']]

zoo_predators = []
for i, link in enumerate(food_web_settings):
    if link['prey'] in zoo_names:
        zoo_predators.append(link['predator'])

n_links = len(food_web_settings)
coord = [f"{link['predator']}_{link['prey']}" for link in food_web_settings]
coord_zoo = [
    f"{link['predator']}_{link['prey']}" for link in food_web_settings if link['prey'] in zoo_names
]
masses = {f.name: f.mass for f in F.fish}

predator_list = [link['predator'] for link in food_web_settings]
prey_list = [link['prey'] for link in food_web_settings]

pred_list_prey = {
    pred: [prey for i, prey in enumerate(prey_list) if predator_list[i] == pred]
    for pred in np.unique(predator_list)
}

prey_list_pred = {
    prey: [pred for i, pred in enumerate(predator_list) if prey_list[i] == prey]
    for prey in np.unique(prey_list)
}


def test_food_web_init_1():
    """ensure food_web conforms"""
    assert set(F.food_web.__dict__.keys()) == {
        'fish',
        'fish_names',
        'n_links',
        'preference',
        'ndx_prey',
        'prey_ndx_pred',
        'pred_prey_preference',
        '_index',
        'i_fish',
        'prey_link_ndx',
        'pred_ndx_prey',
        'prey_obj',
        'zoo_names',
        'pred_link_ndx',
        'predator_obj',
        'pred_prey_func_type',
    }

    assert F.food_web.n_links == n_links
    assert [o.name for o in F.food_web.predator_obj] == predator_list
    assert [o.name for o in F.food_web.prey_obj] == prey_list

    for key in ['predator_obj', 'prey_obj']:
        expected_types = (
            [
                ecosystem.fish_type,
                ecosystem.zooplankton_type,
                ecosystem.benthic_prey_type,
            ]
            if key == 'prey_obj'
            else [ecosystem.fish_type]
        )
        assert len(F.food_web.__dict__[key]) == len(food_web_settings)
        for i in range(len(F.food_web.__dict__[key])):
            assert any([isinstance(F.food_web.__dict__[key][i], t) for t in expected_types])

    for link in food_web_settings:
        pred = link['predator']
        prey = link['prey']
        assert (
            F.food_web.pred_prey_preference.sel(predator=pred, prey=prey)
            == link['encounter_parameters']['preference']
        )

    print(F.food_web.predator_obj)
    assert all([isinstance(fish, ecosystem.fish_type) for fish in F.food_web.predator_obj])


def test_food_web_init_2():
    """test encounter and consumption objects"""

    expected_types = dict(
        predator=[ecosystem.fish_type],
        prey=[ecosystem.fish_type, ecosystem.zooplankton_type, ecosystem.benthic_prey_type],
    )
    for link in F.food_web:
        assert any([isinstance(link.predator, t) for t in expected_types['predator']])
        assert any([isinstance(link.prey, t) for t in expected_types['prey']])


def test_food_web_init_3():
    """confirm the pred/prey indexes are correct"""

    group_func_type = np.array([o.functional_type for o in F.member_obj_list], dtype=object)
    for pred, ndx in F.food_web.pred_ndx_prey.items():
        assert (F.biomass.group[ndx] == pred_list_prey[pred]).all()

        assert (F.food_web.pred_prey_func_type[pred] == group_func_type[ndx]).all()

        assert F.food_web.pred_link_ndx[pred] == [
            i for i in range(n_links) if predator_list[i] == pred
        ]

    for prey, ndx in F.food_web.prey_ndx_pred.items():
        assert (F.biomass.group[ndx] == prey_list_pred[prey]).all()
        assert F.food_web.prey_link_ndx[prey] == [i for i in range(n_links) if prey_list[i] == prey]


def test_missing_fish():
    settings_dict_def_bad = feisty.settings.get_defaults()
    settings_dict_def_bad['food_web'] = [d for d in food_web_settings if d['predator'] != 'Md']
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=settings_dict_def_bad,
            fish_ic_data=fish_ic_data,
        )


def test_nonfish_predator():
    sd_bad = feisty.settings.get_defaults()
    sd_bad['food_web'][-1]['predator'] = zoo_names[0]
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_bad,
            fish_ic_data=fish_ic_data,
        )


def test_pred_ndx_prey_filt():
    """verify the "_get_df_predator" method"""

    for pred, prey_list_check in all_prey.items():

        ndx = F.food_web._pred_ndx_prey_filt(pred)
        prey_list = F.biomass.group.values[ndx]
        assert (np.array(prey_list) == np.array(prey_list_check)).all()

        for (
            prey_functional_type_key,
            prey_functional_type,
        ) in ecosystem.functional_types.items():
            ndx = F.food_web._pred_ndx_prey_filt(pred, set([prey_functional_type]))

            if not ndx:
                continue
            prey_list = F.biomass.group.values[ndx]
            prey_list_check_filt = [
                p for p in prey_list_check if fish_func_type[p] == prey_functional_type_key
            ]
            assert (prey_list == prey_list_check_filt).all()


def test_duplicated_link_fails():
    settings_dict_def_bad = feisty.settings.get_defaults()
    settings_dict_def_bad['food_web'].append(settings_dict_def_bad['food_web'][0])
    with pytest.raises(AssertionError):
        feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=settings_dict_def_bad,
            fish_ic_data=fish_ic_data,
        )


def test_get_prey_biomass():
    """verify the `get_prey_biomass` method"""

    # generate random biomass data
    data = xr.full_like(F.biomass, fill_value=0.0)
    data.data[:, :] = np.random.rand(*data.shape)

    # set the feisty_instance biomass array to these random values
    # save the old values
    zoo_data_prior = F.biomass.isel(group=F.ndx_zoo).data
    fish_data_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data

    F.set_zoo_biomass(data.isel(group=F.ndx_zoo))
    F.set_fish_biomass(data.isel(group=F.ndx_fish))
    F.set_benthic_prey_biomass(data.isel(group=F.ndx_benthic_prey))

    # ensure that the prey biomass returned matchs that input
    for pred_obj, prey_obj in zip(F.food_web.predator_obj, F.food_web.prey_obj):
        pred = pred_obj.name
        prey_list_check = all_prey[pred]

        da = F.food_web.get_prey_biomass(F.biomass, pred)
        check_value = data.sel(group=prey_list_check).sum('group')
        assert (check_value == da).all()

        # ensure that if *all* functional types are passed in, that the sum is equivalent
        da = F.food_web.get_prey_biomass(
            F.biomass,
            pred,
            prey_functional_type=list(ecosystem.functional_types.values()),
        )
        assert (check_value == da).all()

        da = F.food_web.get_prey_biomass(F.biomass, pred, apply_preference=True)
        check_value = (
            data.sel(group=prey_list_check) * xr.DataArray(preference[pred], dims=('group'))
        ).sum('group')
        assert (check_value == da).all()

        # ensure that this works for a restricted functional type
        for prey_functional_type in ecosystem.functional_types.values():

            prey_list_check_filt = [
                p
                for p in prey_list_check
                if ecosystem.functional_types[fish_func_type[p]] == prey_functional_type
            ]
            da = F.food_web.get_prey_biomass(
                F.biomass,
                pred,
                prey_functional_type=set([prey_functional_type]),
            )
            assert (data.sel(group=prey_list_check_filt).sum('group') == da).all()

        # check that pelagic functional type sums work
        prey_functional_type_keys = model_settings['pelagic_functional_type_keys']
        prey_list_check_filt = [
            p for p in prey_list_check if fish_func_type[p] in prey_functional_type_keys
        ]
        da = F.food_web.get_prey_biomass(
            F.biomass,
            pred,
            prey_functional_type=[ecosystem.functional_types[p] for p in prey_functional_type_keys],
        )
        assert (data.sel(group=prey_list_check_filt).sum('group') == da).all()

        # check that demersal functional type sums work
        prey_functional_type_keys = model_settings['demersal_functional_type_keys']
        prey_list_check_filt = [
            p for p in prey_list_check if fish_func_type[p] in prey_functional_type_keys
        ]
        da = F.food_web.get_prey_biomass(
            F.biomass,
            pred,
            prey_functional_type=[ecosystem.functional_types[p] for p in prey_functional_type_keys],
        )
        assert (data.sel(group=prey_list_check_filt).sum('group') == da).all()

    # put it back
    F.set_zoo_biomass(zoo_data_prior)
    F.set_fish_biomass(fish_data_prior)
    F.set_benthic_prey_biomass(benthic_data_prior)


def test_get_prey_biomass_dne():
    """ensure that we can't get biomass for non-existence functional type"""
    with pytest.raises(AssertionError):
        F.food_web.get_prey_biomass(
            F.biomass,
            settings_dict_def['food_web'][0]['predator'],
            prey_functional_type=['this-is-not-a-valid-functional-type'],
        )


def test_get_consumption_bad_args():
    # no index args
    data = xr.full_like(F.tendency_data.consumption_rate_link, fill_value=0.0)
    with pytest.raises(AssertionError):
        F.food_web.get_consumption(data)

    # bad dims
    data = xr.full_like(F.tendency_data.ingestion_rate, fill_value=0.0)
    with pytest.raises(AssertionError):
        F.food_web.get_consumption(data, prey='zooplankton')


def test_get_consumption_none_existent_predprey():
    data = xr.full_like(F.tendency_data.consumption_rate_link, fill_value=0.0)
    assert F.food_web.get_consumption(data, predator='big-fat-tuna') is None
    assert F.food_web.get_consumption(data, prey='small-stinky-sardine') is None


# def test_compute_consumption_zero_preference():
#     sd = feisty.settings.get_defaults()
#     for i in range(len(sd['food_web'])):
#         sd['food_web'][i]['encounter_parameters']['preference'] = 0.0
#     fw = feisty.core.ecosystem.food_web(sd['food_web'], F.member_obj_list)
#     fw._compute_encounter(F.biomass, F.tendency_data.T_habitat, F.tendency_data.t_frac_pelagic)
#     assert (fw.encounter == 0.0).all()
#


def test_get_consumption():
    data = xr.full_like(F.tendency_data.consumption_rate_link, fill_value=0.0)
    data.data[:, :] = np.random.rand(*data.shape)
    F.tendency_data.consumption_rate_link.data[:, :] = data.data

    for pred in predator_list:
        pred_link_ndx = [i for i, link in enumerate(food_web_settings) if link['predator'] == pred]
        pred_prey_list = [
            prey_list[i] for i, link in enumerate(food_web_settings) if link['predator'] == pred
        ]
        print(pred)
        print(pred_prey_list)
        print()
        da = F.food_web.get_consumption(data, predator=pred)

        assert (da.data == data.isel(feeding_link=pred_link_ndx).data).all()
        assert da.dims == ('group', 'X')
        assert (da.group == pred_prey_list).all()
        assert da.shape == (len(pred_link_ndx), NX)
        print()

        for prey in pred_prey_list:
            pred_link_ndx = [
                i
                for i, link in enumerate(food_web_settings)
                if link['predator'] == pred and link['prey'] == prey
            ]
            da = F.food_web.get_consumption(data, predator=pred, prey=prey)
            print(da)
            assert (da.data == data.isel(feeding_link=pred_link_ndx).data).all()
            assert da.dims == ('feeding_link', 'X')
            assert da.shape == (len(pred_link_ndx), NX)

    for prey in prey_list:
        prey_link_ndx = [i for i, link in enumerate(food_web_settings) if link['prey'] == prey]

        pred_list = [link['predator'] for link in food_web_settings if link['prey'] == prey]
        da = F.food_web.get_consumption(data, prey=prey)
        assert (da.data == data.isel(feeding_link=prey_link_ndx).data).all()
        assert da.dims == ('group', 'X')
        assert (da.group == pred_list).all()
        assert da.shape == (len(prey_link_ndx), NX)


def test_compute_feeding_1():
    """run thru the feeding computation machinery"""

    F.gcm_state.update(
        T_pelagic=10.0,
        T_bottom=5.0,
        poc_flux=0.0,
    )

    # set the feisty_instance biomass array to these random values
    zoo_data_prior = F.biomass.isel(group=F.ndx_zoo).data
    fish_data_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data

    # generate random biomass data
    data = xr.full_like(F.biomass, fill_value=0.0)
    data.data[:, :] = np.ones(data.shape)
    F.set_zoo_biomass(data.isel(group=F.ndx_zoo))
    F.set_fish_biomass(data.isel(group=F.ndx_fish))
    F.set_benthic_prey_biomass(data.isel(group=F.ndx_benthic_prey))

    F._compute_t_frac_pelagic(reset=True)
    F._compute_temperature()
    F._compute_encounter()
    F._compute_consumption()

    ds = F.tendency_data

    # check that array's conform
    predator = [link['predator'] for link in food_web_settings]

    for pred in predator:
        pred_link_ndx = [i for i, link in enumerate(food_web_settings) if link['predator'] == pred]
        assert (
            ds.encounter_rate_total
            == ds.encounter_rate_link.isel(feeding_link=pred_link_ndx).sum('feeding_link')
        ).all()

    # regression test (not working yet as preferences are random)
    # datafile = f"{path_to_here}/data/food_web_check.nc"
    # with xr.open_dataset(datafile) as ds_expected:
    #    xr.testing.assert_allclose(ds, ds_expected)

    # ensure that the biomass of zoo predators is returned correctly
    for zoo_i in zoo_names:
        biomass_zoo_pred = F.food_web._get_biomass_zoo_pred(F.biomass, zoo_i)
        xr.testing.assert_identical(
            biomass_zoo_pred, F.biomass.isel(group=F.food_web.prey_ndx_pred[zoo_i])
        )

        # ensure that zoo consumption is zoo consumption
        ndx = [i for i, link in enumerate(food_web_settings) if link['prey'] == zoo_i]
        consumption_zoo = F.food_web.get_consumption(ds.consumption_rate_link, prey=zoo_i)
        np.array_equal(consumption_zoo.data, ds.consumption_rate_link.isel(feeding_link=ndx).data)

    assert 'group' in consumption_zoo.coords
    assert 'group' in consumption_zoo.indexes
    assert 'feeding_link' not in consumption_zoo.coords
    assert (consumption_zoo.group == zoo_predators).all()

    # F.food_web._rescale_consumption(F.biomass, zoo_mortality=F.zoo_mortality)
    # assert 0
    # put it back
    F.set_zoo_biomass(zoo_data_prior)
    F.set_fish_biomass(fish_data_prior)
    F.set_benthic_prey_biomass(benthic_data_prior)


def test_rescale_zoo_consumption():
    data = xr.full_like(F.tendency_data.consumption_rate_link, fill_value=0.0)
    data.data[:, :] = np.ones(data.shape) * 10.0

    zoo_mortality_data = np.ones((n_zoo, NX))

    F.tendency_data.consumption_rate_link[:, :] = data
    F.set_zoo_mortality(zoo_mortality_data * 0.0)

    for i, link in enumerate(F.food_web):
        assert (F.tendency_data.consumption_rate_link[i, :] == data[i, :]).all()

    F._compute_rescale_zoo_consumption()

    for i, link in enumerate(F.food_web):
        if link.prey.is_zooplankton:
            assert (F.tendency_data.consumption_rate_link[i, :] == 0.0).all()
