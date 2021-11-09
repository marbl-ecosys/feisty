import numpy as np
import pytest
import xarray as xr

import feisty
import feisty.fish_mod as fish_mod
import feisty.settings as settings

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

all_prey, preference = conftest.get_all_prey_and_preference(settings_dict_def)
fish_func_type = conftest.get_fish_func_type(settings_dict_def)


@pytest.mark.parametrize(
    'Tp, Tb, t_frac_pelagic, expected',
    [
        (10.0, 1.0, 1.0, 10.0),
        (10.0, 1.0, 0.5, 5.5),
        (10.0, 1.0, 0.0, 1.0),
    ],
)
def test_t_weighted_mean_temp(Tp, Tb, t_frac_pelagic, expected):
    assert fish_mod.t_weighted_mean_temp(Tp, Tb, t_frac_pelagic) == expected


def test_t_frac_pelagic():
    # set the feisty_instance biomass array to these random values
    zoo_data_prior = F.biomass.isel(group=F.ndx_zoo).data
    fish_data_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data

    # generate random biomass data
    data = xr.full_like(F.biomass, fill_value=0.0)
    data.data[:, :] = np.random.rand(*data.shape)
    F.set_zoo_biomass(data.isel(group=F.ndx_zoo))
    F.set_fish_biomass(data.isel(group=F.ndx_fish))
    F.set_benthic_prey_biomass(data.isel(group=F.ndx_benthic_prey))

    F._compute_t_frac_pelagic()

    pelagic_functional_types = model_settings['pelagic_functional_types']
    demersal_functional_types = model_settings['demersal_functional_types']
    ocean_depth = domain_dict['depth_of_seafloor']
    PI_be_cutoff = model_settings['PI_be_cutoff']

    for i, fish in enumerate(F.fish):
        pred = fish.name
        t_frac_pelagic = np.array(fish.t_frac_pelagic.values)

        if fish.pelagic_demersal_coupling:
            prey_list_check = all_prey[pred]

            prey_list_check_pelagic = []
            prey_list_check_demersal = []
            preference_check_pelagic = []
            preference_check_demersal = []
            for j, p in enumerate(prey_list_check):
                if fish_func_type[p] in pelagic_functional_types:
                    prey_list_check_pelagic.append(p)
                    preference_check_pelagic.append(preference[pred][j])

                elif fish_func_type[p] in demersal_functional_types:
                    prey_list_check_demersal.append(p)
                    preference_check_demersal.append(preference[pred][j])

            if fish._pdc_apply_pref:
                prey_pelagic = (
                    data.sel(group=prey_list_check_pelagic)
                    * xr.DataArray(preference_check_pelagic, dims=('group'))
                ).sum('group')
                prey_demersal = (
                    data.sel(group=prey_list_check_demersal)
                    * xr.DataArray(preference_check_demersal, dims=('group'))
                ).sum('group')
            else:
                prey_pelagic = data.sel(group=prey_list_check_pelagic).sum('group')
                prey_demersal = data.sel(group=prey_list_check_demersal).sum('group')

                da = F.food_web.get_prey_biomass(
                    F.biomass,
                    pred,
                    prey_functional_type=[
                        fish_mod.functional_types[p] for p in pelagic_functional_types
                    ],
                )
                assert (da.data == prey_pelagic).all()

                da = F.food_web.get_prey_biomass(
                    F.biomass,
                    pred,
                    prey_functional_type=[
                        fish_mod.functional_types[p] for p in demersal_functional_types
                    ],
                )
                assert (da.data == prey_demersal).all()

            t_frac_pelagic = np.where(
                ocean_depth < PI_be_cutoff,
                prey_pelagic.data / (prey_pelagic.data + prey_demersal.data),
                1.0,
            )
        # we should have successfully reconstructed the prey ratio
        assert (F.tendency_data.t_frac_pelagic.data[i, :] == t_frac_pelagic).all()

    # put it back
    F.set_zoo_biomass(zoo_data_prior)
    F.set_fish_biomass(fish_data_prior)
    F.set_benthic_prey_biomass(benthic_data_prior)

    F._compute_t_frac_pelagic(reset=True)
    for i, fish in enumerate(F.fish):
        assert (fish.t_frac_pelagic == fish_settings[i]['t_frac_pelagic_static']).all()


@pytest.mark.weak
def test_update_benthic_prey():
    """test benthic update

    Add regression data
    """
    F._update_benthic_biomass()


def test_compute_metabolism():
    F.gcm_state.update(
        T_pelagic=10.0,
        T_bottom=5.0,
        poc_flux=0.0,
    )
    F._compute_t_frac_pelagic(reset=True)
    F._compute_temperature()
    F._compute_metabolism()

    datafile = f'{conftest.path_to_here}/data/metabolism_check.nc'
    with xr.open_dataarray(datafile) as expected:
        xr.testing.assert_allclose(F.tendency_data.metabolism_rate, expected)


def test_compute_ingestion():
    ingestion = xr.full_like(F.food_web.consumption, fill_value=0.0)
    ingestion.data[:, :] = np.random.rand(*ingestion.shape)
    F.food_web.consumption.data[:, :] = ingestion.data

    F._compute_ingestion()

    for fish in F.fish:
        ndx = [i for i, link in enumerate(food_web_settings) if link['predator'] == fish.name]
        assert (
            F.tendency_data.ingestion_rate.sel(fish=fish.name)
            == ingestion.isel(feeding_link=ndx).sum('feeding_link')
        ).all()


def test_compute_predation():
    ingestion = xr.full_like(F.food_web.consumption, fill_value=0.0)
    ingestion.data[:, :] = np.random.rand(*ingestion.shape)
    F.food_web.consumption.data[:, :] = ingestion.data

    biomass = xr.full_like(F.biomass, fill_value=0.0)
    biomass.data[:, :] = np.random.rand(*biomass.shape)
    F.biomass.data[:, :] = biomass.data

    F._compute_predation()

    for fish in F.fish:
        print(fish.name)
        ndx = [i for i, link in enumerate(food_web_settings) if link['prey'] == fish.name]
        if not ndx:
            continue

        pred_list = [link['predator'] for link in food_web_settings if link['prey'] == fish.name]

        consumption = (
            ingestion.isel(feeding_link=ndx)
            .reset_index('feeding_link', drop=True)
            .set_index(feeding_link='predator')
            .rename(feeding_link='group')
        )
        biomass_pred = biomass.sel(group=pred_list)
        biomass_prey = biomass.sel(group=fish.name)
        predation = (consumption * biomass_pred).sum('group')
        assert (consumption.group == pred_list).all()
        assert (consumption.prey == fish.name).all()

        assert (F.tendency_data.predation_flux.sel(fish=fish.name) == predation).all()

        assert (
            F.tendency_data.predation_rate.sel(fish=fish.name) == predation / biomass_prey
        ).all()


@pytest.mark.weak
def test_compute_mortality():
    """test mortality

    Add regression check data.
    """
    F.gcm_state.update(
        T_pelagic=10.0,
        T_bottom=5.0,
        poc_flux=0.0,
    )
    F._compute_t_frac_pelagic(reset=True)
    F._compute_temperature()
    F._compute_mortality()

    assert (F.tendency_data.mortality_rate == 0.1 / 365.0).all()

    for mortality_type in fish_mod._mortality_type_keys:
        print(f'testing {mortality_type}')
        sd_mort = feisty.settings.get_defaults()
        for i in range(len(sd_mort['fish'])):
            sd_mort['fish'][i]['mortality_type'] = mortality_type

        Fprime = feisty.feisty_instance_type(
            domain_dict=domain_dict,
            settings_dict=sd_mort,
        )
        Fprime._compute_mortality()


@pytest.mark.weak
def test_compute_nu():
    F._compute_energy_avail()


@pytest.mark.weak
def test_compute_gamma():
    F._compute_growth()


@pytest.mark.weak
def test_compute_reproduction():
    F._compute_reproduction()


@pytest.mark.weak
def test_compute_recruitment():
    F._compute_recruitment()


@pytest.mark.weak
def test_compute_total_tendency():
    F._compute_total_tendency()


@pytest.mark.weak
def test_compute_fish_catch():
    F._compute_fish_catch()


@pytest.mark.weak
def test_compute_tendencies():
    fish_biomass_prior = F.biomass.isel(group=F.ndx_fish).data
    benthic_data_prior = F.biomass.isel(group=F.ndx_benthic_prey).data
    zooplankton_prior = F.biomass.isel(group=F.ndx_zoo).data

    fish_biomass = np.random.rand(n_fish, NX)
    benthic_prey_biomass = np.random.rand(n_benthic_prey, NX)
    zooplankton_data = np.random.rand(n_zoo, NX)
    zoo_mortality_data = np.random.rand(n_zoo, NX)

    T_pelagic = 23.0
    T_bottom = 3.0
    poc_flux = 1.0

    F.compute_tendencies(
        fish_biomass,
        benthic_prey_biomass,
        zooplankton_data,
        zoo_mortality_data,
        T_pelagic=T_pelagic,
        T_bottom=T_bottom,
        poc_flux=poc_flux,
    )

    F.set_fish_biomass(fish_biomass_prior)
    F.set_zoo_biomass(zooplankton_prior)
    F.set_benthic_prey_biomass(benthic_data_prior)
    F.set_zoo_mortality(zoo_mortality_data * 0.0)
