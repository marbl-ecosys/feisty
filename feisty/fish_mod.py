import numpy as np
import xarray as xr

from . import constants, domain

functional_types = {}

_pdc_type_keys = []
_pdc_apply_pref_func_types = []
_pelagic_functional_types = []
_pelagic_functional_type_keys = set()
_demersal_functional_types = []
_demersal_functional_type_keys = set()

_size_class_masses = {}
_size_class_bnds_ratio = {}
_PI_be_cutoff = None

_mortality_type_keys = [
    'none',
    'constant',
    'Hartvig',
    'Mizer',
    'Jennings & Collingridge',
    'Peterson & Wrob',
    'temperature-dependent',
    'weight-dependent',
]
_mortality_types = {k: i for i, k in enumerate(_mortality_type_keys)}


def init_module_variables(
    size_class_bounds,
    functional_type_names,
    PI_be_cutoff,
    pelagic_demersal_coupling_types,
    pelagic_demersal_coupling_apply_pref_types,
    pelagic_functional_types,
    demersal_functional_types,
):
    global functional_types
    global _size_class_masses
    global _size_class_bnds_ratio
    global _PI_be_cutoff
    global _pdc_type_keys
    global _pdc_apply_pref_func_types
    global _pelagic_functional_types
    global _demersal_functional_types
    global _pelagic_functional_type_keys
    global _demersal_functional_type_keys

    for name, size_bounds in size_class_bounds.items():
        _size_class_masses[name] = np.power(10.0, np.log10(size_bounds).mean())
        _size_class_bnds_ratio[name] = size_bounds[0] / size_bounds[1]

    for i, name in enumerate(functional_type_names):
        functional_types[name] = i

    # check inputs
    assert not set(pelagic_demersal_coupling_types) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_demersal_coupling_types` list: {pelagic_demersal_coupling_types}'

    assert not set(pelagic_demersal_coupling_apply_pref_types) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_demersal_coupling_apply_pref_types` list: {pelagic_demersal_coupling_apply_pref_types}'

    assert not set(pelagic_demersal_coupling_apply_pref_types) - set(
        pelagic_demersal_coupling_types
    ), f'pelagic_demersal_coupling_apply_pref_types specifies types not found in pelagic_demersal_coupling_types: {pelagic_demersal_coupling_apply_pref_types}'

    assert not set(pelagic_functional_types) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_functional_types` list: {pelagic_functional_types}'

    assert not set(demersal_functional_types) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `demersal_functional_types` list: {demersal_functional_types}'

    assert not set(demersal_functional_types).intersection(
        set(pelagic_functional_types)
    ), f'unknown functional type specified in `demersal_functional_types` list: {demersal_functional_types}'

    # make assignments
    _pdc_type_keys = list(pelagic_demersal_coupling_types)
    _pelagic_functional_types = set([functional_types[f] for f in pelagic_functional_types])
    _pelagic_functional_type_keys = set(pelagic_functional_types)
    _demersal_functional_types = set([functional_types[f] for f in demersal_functional_types])
    _demersal_functional_type_keys = set(demersal_functional_types)
    _pdc_apply_pref_func_types = [
        functional_types[k] for k in pelagic_demersal_coupling_apply_pref_types
    ]
    _PI_be_cutoff = PI_be_cutoff


# types
class fish_type(object):
    def __init__(
        self,
        name,
        size_class,
        functional_type,
        t_frac_pelagic_static,
        pelagic_demersal_coupling,
        harvest_selectivity,
        energy_frac_somatic_growth,
        kt=0.0855,
        amet=4.0,
        bpow=0.175,
        mortality_type='constant',
        mortality_coeff_per_yr=0.1,
        assim_efficiency=0.7,
    ):
        """
        Paramterization of fish functional types.

        Parameters
        ----------

        functional_type: string
          The functional type of this group, i.e., one of the following:
             ["forage", "piscivore", "demersal"]

        mass : float
          Individual Mass (g) = geometric mean

        kt : float
          Coeff on met T-dep fn (orig 0.063) %0.0855

        amet : float
          Coeff on metabolism.

        bpow : float
          power on metab fn

        """
        assert size_class in _size_class_masses, f'Unknown size class {size_class}'
        assert functional_type in functional_types, f'Unknown functional type: {functional_type}'
        assert mortality_type in _mortality_type_keys, f'Unknown mortality type: {mortality_type}'

        assert (0.0 <= harvest_selectivity) and (
            harvest_selectivity <= 1.0
        ), 'harvest_selectivity must be between 0. and 1.'
        assert (0.0 <= energy_frac_somatic_growth) and (
            energy_frac_somatic_growth <= 1.0
        ), 'energy_frac_somatic_growth must be between 0. and 1.'
        assert (0.0 <= assim_efficiency) and (
            assim_efficiency <= 1.0
        ), 'assim_efficiency must be between 0. and 1.'

        self.name = name
        self.functional_type_key = functional_type
        self.functional_type = functional_types[functional_type]
        self.size_class = size_class

        self.mass = _size_class_masses[size_class]

        self.kt = kt
        self.amet = amet
        self.bpow = bpow

        self.t_frac_pelagic_static = t_frac_pelagic_static
        self.pelagic_demersal_coupling = pelagic_demersal_coupling
        if self.pelagic_demersal_coupling:
            assert self.functional_type in [
                functional_types[t] for t in _pdc_type_keys
            ], f"pelagic-demersal coupling not defined for '{functional_type}' functional type"

        # initialize memory for result
        self.t_frac_pelagic = domain.init_array(
            name=f'{self.name}_t_frac_pelagic',
            constant=self.t_frac_pelagic_static,
        )

        self.harvest_selectivity = harvest_selectivity
        self.energy_frac_somatic_growth = energy_frac_somatic_growth
        self.mortality_type = _mortality_types[mortality_type]
        self.mortality_coeff = mortality_coeff_per_yr / 365.0
        self.assim_efficiency = assim_efficiency

    def __repr__(self):
        return f'{self.name}: {self.size_class} {self.functional_type_key}'

    @property
    def _pdc_apply_pref(self):
        return True if self.functional_type in _pdc_apply_pref_func_types else False


class zooplankton_type(object):
    """Data structure containing zooplankton parameters."""

    def __init__(self, name):
        self.name = name
        self.functional_type_key = 'zooplankton'
        self.functional_type = functional_types['zooplankton']


class benthic_prey_type(object):
    """Data structure containing benthic prey parameters."""

    def __init__(
        self,
        name,
        benthic_efficiency,
        carrying_capacity=0.0,
    ):
        self.name = name
        self.functional_type_key = 'benthic_prey'
        self.functional_type = functional_types['benthic_prey']
        self.benthic_efficiency = benthic_efficiency

        self.carrying_capacity = carrying_capacity
        self.lcarrying_capacity = carrying_capacity == 0.0


class fishing(object):
    """Data structure containing fishing parameters"""

    def __init__(self, fishing_rate_per_year):
        self.fishing_rate = domain.init_array(
            name='fishing_rate',
            constant=fishing_rate_per_year / 365.0,
            attrs={'long_name': 'Imposed fishing rate', 'units': '1/d'},
        )


def is_demersal(key):
    """Return `True` if key is a demersal functional type"""
    return key in _demersal_functional_type_keys


def t_weighted_mean_temp(Tp, Tb, t_frac_pelagic):
    """compute weighted-mean temperature"""
    return (Tp * t_frac_pelagic) + (Tb * (1.0 - t_frac_pelagic))


def compute_t_frac_pelagic(da, fish_list, biomass, food_web, reset=False):
    """return the fraction of time spent in the pelagic"""

    for i, fish in enumerate(fish_list):
        if reset:
            da.data[i, :] = fish.t_frac_pelagic_static

        elif fish.pelagic_demersal_coupling:
            prey_pelagic = food_web.get_prey_biomass(
                biomass,
                fish.name,
                prey_functional_type=_pelagic_functional_types,
                apply_preference=fish._pdc_apply_pref,
            )
            prey_demersal = food_web.get_prey_biomass(
                biomass,
                fish.name,
                prey_functional_type=_demersal_functional_types,
                apply_preference=fish._pdc_apply_pref,
            )

            da.data[i, :] = xr.where(
                domain.ocean_depth < _PI_be_cutoff,
                prey_pelagic / (prey_pelagic + prey_demersal),
                1.0,
            )


def compute_metabolism(metabolism_rate, fish_list, T_habitat):
    """
    Compute metabolic rate.
    Tp: pelagic temp
    Tb: bottom temp
    t_frac_pelagic: frac pelagic time
    mass: ind weight of size class
    fcrit: feeding level to meet resting respiration rate
    cmax: max consumption rate
    U: swimming speed
    """

    for i, fish in enumerate(fish_list):
        # Metabolism with its own coeff, temp-sens, mass-sens
        metabolism_rate[i, :] = (
            np.exp(fish.kt * (T_habitat[i, :] - 10.0)) * fish.amet * fish.mass ** (-fish.bpow)
        ) / 365.0


def compute_ingestion(ingestion_rate, food_web):
    """Compute ingestion"""
    for i, name in enumerate(food_web.fish_names):
        ingestion_rate[i, :] = food_web.get_consumption(predator=name).sum('group')


def compute_predation(predation_flux, food_web, biomass):
    """Compute predation"""
    for i, name in enumerate(food_web.fish_names):
        # not eaten?
        if name not in food_web.prey_ndx_pred:
            continue

        ndx = food_web.prey_ndx_pred[name]
        predation_flux[i, :] = (food_web.get_consumption(prey=name) * biomass[ndx, :]).sum('group')


def compute_benthic_biomass_update(da, benthic_prey_list, biomass, food_web, poc_flux):
    """
    bio_in = benthic biomass
    det = poc_flux flux to bottom (g/m2/d)
    con = biomass specific consumption rate by MD & LD
    bio = biomass of MD & LD
    """

    for i, benthic_prey in enumerate(benthic_prey_list):
        # eaten = consumption * biomass_pred
        # pred = sum(eaten, 2)
        biomass_bent = biomass.sel(group=benthic_prey.name)
        predation = (
            biomass.isel(group=food_web.prey_ndx_pred[benthic_prey.name])
            * food_web.get_consumption(prey=benthic_prey.name)
        ).sum('group')

        # Needs to be in units of per time (g/m2/d) * (g/m2)
        growth = benthic_prey.benthic_efficiency * poc_flux

        if not benthic_prey.lcarrying_capacity:  # no carrying capacity
            da.data[i, :] = biomass_bent + growth - predation
        else:
            # logistic
            da.data[i, :] = (
                biomass_bent
                + growth * (1.0 - biomass_bent / benthic_prey.carrying_capacity)
                - predation
            )

    da.data[:, :] = np.where(da.data < 0.0, constants.eps, da.data)


def natural_mortality(mortality_rate, fish_list, T_habitat):
    """
    Temp-dep natural mortality
    Tp: pelagic temp
    Tb: bottom temp
    t_frac_pelagic: frac pelagic time
    """

    for i, fish in enumerate(fish_list):

        if fish.mortality_type == _mortality_types['none']:
            mortality_rate[i, :] = 0.0

        elif fish.mortality_type == _mortality_types['constant']:
            mortality_rate[i, :] = fish.mortality_coeff

        elif fish.mortality_type == _mortality_types['Hartvig']:
            mortality_rate[i, :] = (
                np.exp(0.063 * (T_habitat[i, :] - 10.0)) * 0.84 * fish.mass ** (-0.25) / 365.0
            )

        elif fish.mortality_type == _mortality_types['Mizer']:
            mortality_rate[i, :] = (
                np.exp(0.063 * (T_habitat[i, :] - 10.0)) * 3.0 * fish.mass ** (-0.25) / 365.0
            )

        elif fish.mortality_type == _mortality_types['Jennings & Collingridge']:
            # TODO: clean up here
            temp2 = T_habitat[i, :] + 273.0
            Tref = 283.0
            E = 0.6
            k = 8.62e-5
            tfact = np.exp((-1 * E / k) * ((1.0 / temp2) - (1.0 / Tref)))
            mortality_rate[i, :] = tfact * 0.5 * fish.mass ** (-0.33) / 365.0

        elif fish.mortality_type == _mortality_types['Peterson & Wrob']:
            # Peterson & Wroblewski (daily & uses dry weight)
            mortality_rate[i, :] = (
                np.exp(0.063 * (T_habitat[i, :] - 15.0)) * 5.26e-3 * (fish.mass / 9.0) ** (-0.25)
            )

        elif fish.mortality_type == _mortality_types['temperature-dependent']:
            mortality_rate[i, :] = np.exp(0.063 * (T_habitat[i, :] - 10.0)) * fish.mortality_coeff

        elif fish.mortality_type == _mortality_types['weight-dependent']:
            mortality_rate[i, :] = 0.5 * fish.mass ** (-0.25) / 365.0

        else:
            raise ValueError(f'unknown mortality type {fish.mortality_type}')


def compute_energy_avail(energy_avail_rate, ingestion_rate, metabolism_rate, fish_list):
    """Compute energy available for growth (nu)"""

    for i, fish in enumerate(fish_list):
        energy_avail_rate[i, :] = (ingestion_rate[i, :] * fish.assim_efficiency) - metabolism_rate[
            i, :
        ]


def compute_growth(
    growth_rate, energy_avail_rate, predation_rate, mortality_rate, fish_catch_rate, fish_list
):
    """Compute energy available for somatic growth.
    nmort = natural mortality rate
    Frate = fishing mortality rate
    d = predation loss
    selec = harvested selectivity (adults 100%, juveniles 10%)
    """

    for i, fish in enumerate(fish_list):
        Z = _size_class_bnds_ratio[fish.size_class]

        death = predation_rate[i, :] + mortality_rate[i, :] + fish_catch_rate[i, :]
        somatic_growth_potential = fish.energy_frac_somatic_growth * energy_avail_rate[i, :]

        gg = (somatic_growth_potential - death) / (
            1.0 - (Z ** (1.0 - (death / somatic_growth_potential)))
        )
        growth_rate[i, :] = xr.where(gg < energy_avail_rate[i, :], gg, energy_avail_rate[i, :])
        lndx = np.isnan(gg) | (gg < 0)
        growth_rate[i, lndx] = 0.0


def compute_reproduction(reproduction_rate, growth_rate, energy_avail_rate, fish_list):
    """
    %%% BIOMASS MADE FROM REPRODUCTION
    function [gamma, nu, rep] = sub_rep(NX,gamma,nu,K)
    %nu: energy for growth or spawning
    %K: proportion allocated to growth
    % NOTE: Still never going to accumulate biomass as muscle tissue
    """
    for i, fish in enumerate(fish_list):

        if fish.energy_frac_somatic_growth == 1.0:
            reproduction_rate[i, :] = 0.0
        else:
            # energy available
            rho = xr.where(
                energy_avail_rate[i, :] > 0.0,
                (1.0 - fish.energy_frac_somatic_growth) * energy_avail_rate[i, :],
                0.0,
            )
            # add what would be growth to next size up as repro
            reproduction_rate[i, :] = rho + growth_rate[i, :]
            growth_rate[i, :] = 0.0


class reproduction_routing(object):
    """Data structure to store reproduction routing information."""

    def __init__(self, routing_settings, fish_list, all_groups):

        if isinstance(all_groups, xr.DataArray):
            all_groups = all_groups.values
        elif isinstance(all_groups, list):
            all_groups = np.array(all_groups, dtype=object)

        self._n_links = len(routing_settings)
        self._index = 0

        self.ndx_from = [np.where(link['from'] == all_groups)[0][0] for link in routing_settings]

        fish_names = np.array([f.name for f in fish_list], dtype=object)
        self.i_fish = [np.where(link['to'] == fish_names)[0][0] for link in routing_settings]

        self.is_larval = []
        self.efficiency = []
        for link in routing_settings:
            if 'is_larval' in link:
                self.is_larval.append(link['is_larval'])
            else:
                self.is_larval.append(False)

            if 'efficiency' in link:
                self.efficiency.append(link['efficiency'])
            else:
                self.efficiency.append(None)
        assert not any(
            [il and e is None for il, e in zip(self.is_larval, self.efficiency)]
        ), "reproduction routing with 'is_larval = True' requires 'efficiency' parameter to be set."

    def __len__(self):
        return self._n_links

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self):
            raise StopIteration
        i = self._index
        self._index += 1
        return reproduction_link(
            self.ndx_from[i], self.i_fish[i], self.is_larval[i], self.efficiency[i]
        )


class reproduction_link(object):
    """Data structure with the information pertaining to a specific link in
    reproduction routing list."""

    def __init__(self, ndx_from, i_fish, is_larval, efficiency):
        assert np.isscalar(ndx_from)
        assert np.isscalar(i_fish)
        assert np.isscalar(is_larval)
        assert np.isscalar(efficiency) or efficiency is None
        self.ndx_from = ndx_from
        self.i_fish = i_fish
        self.is_larval = is_larval
        self.efficiency = efficiency


def compute_recruitment(
    recruitment_flux,
    reproduction_rate,
    growth_rate,
    biomass,
    reproduction_routing,
):

    for link in reproduction_routing:
        if link.is_larval:
            recruitment_flux[link.i_fish, :] = (
                link.efficiency
                * reproduction_rate[link.i_fish, :]
                * biomass.isel(group=link.ndx_from)
            )
        else:
            recruitment_flux[link.i_fish, :] = growth_rate[link.i_fish, :] * biomass.isel(
                group=link.ndx_from
            )


def compute_total_tendency(
    total_tendency,
    recruitment_flux,
    energy_avail_rate,
    growth_rate,
    reproduction_rate,
    mortality_rate,
    predation_flux,
    fish_catch_rate,
    biomass,
    fish_list,
):
    """
    function bio_out = sub_update_fi(bio_in,rec,nu,rep,gamma,die,nmort)
    % all inputs except rec & die are in g g-1 d-1; rec & die are g d-1
    % rec = rec from smaller size class = TOTAL biomass gained from
            recruitment
    % energy_avail = energy avail for growth or repro
    % rep = energy lost to egg production
    % gamma = energy lost to maturation to larger size class
    % nmort = natural mortality rate
    % predation = biomass lost to predation
    """
    for i, fish in enumerate(fish_list):
        total_tendency[i, :] = (
            recruitment_flux[i, :]
            + biomass.sel(group=fish.name)
            * (
                (
                    energy_avail_rate[i, :]
                    - reproduction_rate[i, :]
                    - growth_rate[i, :]
                    - mortality_rate[i, :]
                    - fish_catch_rate[i, :]
                )
            )
            - predation_flux[i, :]
        )


def compute_fish_catch(fish_catch_rate, fishing_rate, fish_list):
    """Compute fishing rate.
    %F = fishing rate per day
    %selec = fishery selectivity
    """
    for i, fish in enumerate(fish_list):
        # Linear fishing mortality
        fish_catch_rate[i, :] = fish.harvest_selectivity * fishing_rate[:]
