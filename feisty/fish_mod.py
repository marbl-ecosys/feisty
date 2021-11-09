import numpy as np
import xarray as xr

from . import constants, domain

functional_types = {}

_pdc_type_keys = []
_pdc_apply_pref_func_types = []
pelagic_functional_types = []
_pelagic_functional_type_keys = set()
demersal_functional_types = []
_demersal_functional_type_keys = set()

_size_class_masses = {}
_size_class_bnds_ratio = {}
PI_be_cutoff = None

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
mortality_types = {k: i for i, k in enumerate(_mortality_type_keys)}


def init_module_variables(
    size_class_bounds,
    functional_type_keys,
    benthic_pelagic_depth_cutoff,
    pelagic_demersal_coupling_type_keys,
    pelagic_demersal_coupling_apply_pref_type_keys,
    pelagic_functional_type_keys,
    demersal_functional_type_keys,
):
    global functional_types
    global _size_class_masses
    global _size_class_bnds_ratio
    global PI_be_cutoff
    global _pdc_type_keys
    global _pdc_apply_pref_func_types
    global pelagic_functional_types
    global demersal_functional_types
    global _pelagic_functional_type_keys
    global _demersal_functional_type_keys

    for name, size_bounds in size_class_bounds.items():
        _size_class_masses[name] = np.power(10.0, np.log10(size_bounds).mean())
        _size_class_bnds_ratio[name] = size_bounds[0] / size_bounds[1]

    for i, name in enumerate(functional_type_keys):
        functional_types[name] = i

    # check inputs
    assert not set(pelagic_demersal_coupling_type_keys) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_demersal_coupling_type_keys` list: {pelagic_demersal_coupling_type_keys}'

    assert not set(pelagic_demersal_coupling_apply_pref_type_keys) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_demersal_coupling_apply_pref_type_keys` list: {pelagic_demersal_coupling_apply_pref_type_keys}'

    assert not set(pelagic_demersal_coupling_apply_pref_type_keys) - set(
        pelagic_demersal_coupling_type_keys
    ), f'pelagic_demersal_coupling_apply_pref_types specifies types not found in pelagic_demersal_coupling_type_keys: {pelagic_demersal_coupling_apply_pref_type_keys}'

    assert not set(pelagic_functional_type_keys) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_functional_type_keys` list: {pelagic_functional_type_keys}'

    assert not set(demersal_functional_type_keys) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `demersal_functional_type_keys` list: {demersal_functional_type_keys}'

    assert not set(demersal_functional_type_keys).intersection(
        set(pelagic_functional_type_keys)
    ), f'unknown functional type specified in `demersal_functional_type_keys` list: {demersal_functional_type_keys}'

    # make assignments
    _pdc_type_keys = set(pelagic_demersal_coupling_type_keys)

    _pelagic_functional_type_keys = set(pelagic_functional_type_keys)
    _demersal_functional_type_keys = set(demersal_functional_type_keys)

    pelagic_functional_types = set([functional_types[f] for f in _pelagic_functional_type_keys])

    demersal_functional_types = set([functional_types[f] for f in _demersal_functional_type_keys])

    _pdc_apply_pref_func_types = [
        functional_types[k] for k in pelagic_demersal_coupling_apply_pref_type_keys
    ]
    PI_be_cutoff = benthic_pelagic_depth_cutoff


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
        self.size_class_bnds_ratio = _size_class_bnds_ratio[size_class]

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
        self.mortality_type = mortality_types[mortality_type]
        self.mortality_coeff = mortality_coeff_per_yr / 365.0
        self.assim_efficiency = assim_efficiency

        self.pdc_apply_pref = self.functional_type in _pdc_apply_pref_func_types

    def __repr__(self):
        return f'{self.name}: {self.size_class} {self.functional_type_key}'


def is_demersal(key):
    """Return `True` if key is a demersal functional type"""
    return key in _demersal_functional_type_keys


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


def compute_t_frac_pelagic(da, fish_list, biomass, food_web, reset=False):
    """return the fraction of time spent in the pelagic"""

    for i, fish in enumerate(fish_list):
        if reset:
            da.data[i, :] = fish.t_frac_pelagic_static

        elif fish.pelagic_demersal_coupling:
            prey_pelagic = food_web.get_prey_biomass(
                biomass,
                fish.name,
                prey_functional_type=pelagic_functional_types,
                apply_preference=fish.pdc_apply_pref,
            )
            prey_demersal = food_web.get_prey_biomass(
                biomass,
                fish.name,
                prey_functional_type=demersal_functional_types,
                apply_preference=fish.pdc_apply_pref,
            )

            da.data[i, :] = xr.where(
                domain.ocean_depth < PI_be_cutoff,
                prey_pelagic / (prey_pelagic + prey_demersal),
                1.0,
            )


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
