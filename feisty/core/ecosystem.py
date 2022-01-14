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
_zooplankton_functional_type_keys = set()

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


_fish_defaults = {}
_zooplankton_defaults = {}
_benthic_prey_defaults = {}


def init_module_variables(
    size_class_bounds,
    functional_type_keys,
    benthic_pelagic_depth_cutoff,
    pelagic_demersal_coupling_type_keys,
    pelagic_demersal_coupling_apply_pref_type_keys,
    pelagic_functional_type_keys,
    demersal_functional_type_keys,
    zooplankton_functional_type_keys,
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
    global _zooplankton_functional_type_keys

    for name, size_bounds in size_class_bounds.items():
        _size_class_masses[name] = np.power(10.0, np.log10(size_bounds).mean())
        _size_class_bnds_ratio[name] = size_bounds[0] / size_bounds[1]

    for i, name in enumerate(functional_type_keys):
        functional_types[name] = i

    # check inputs
    assert not set(pelagic_demersal_coupling_type_keys) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `pelagic_demersal_coupling_type_keys` list: {pelagic_demersal_coupling_type_keys}'

    if pelagic_demersal_coupling_apply_pref_type_keys:
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

    assert not set(zooplankton_functional_type_keys) - set(
        functional_types.keys()
    ), f'unknown functional type specified in `zooplankton_functional_type_keys` list: {zooplankton_functional_type_keys}'

    assert not set(demersal_functional_type_keys).intersection(
        set(pelagic_functional_type_keys)
    ), 'functional type overlap'

    assert not set(demersal_functional_type_keys).intersection(
        set(zooplankton_functional_type_keys)
    ), 'functional type overlap'

    assert not set(pelagic_functional_type_keys).intersection(
        set(zooplankton_functional_type_keys)
    ), 'functional type overlap'

    # make assignments
    _pdc_type_keys = set(pelagic_demersal_coupling_type_keys)

    _pelagic_functional_type_keys = set(pelagic_functional_type_keys)
    _demersal_functional_type_keys = set(demersal_functional_type_keys)

    pelagic_functional_types = set([functional_types[f] for f in _pelagic_functional_type_keys])

    demersal_functional_types = set([functional_types[f] for f in _demersal_functional_type_keys])

    _zooplankton_functional_type_keys = set(zooplankton_functional_type_keys)

    if pelagic_demersal_coupling_apply_pref_type_keys:
        _pdc_apply_pref_func_types = [
            functional_types[k] for k in pelagic_demersal_coupling_apply_pref_type_keys
        ]
    else:
        _pdc_apply_pref_func_types = []
    PI_be_cutoff = benthic_pelagic_depth_cutoff


def init_fish_defaults(
    k_metabolism,
    a_metabolism,
    b_metabolism,
    k_encounter,
    a_encounter,
    b_encounter,
    k_consumption,
    a_consumption,
    b_consumption,
    mortality_type,
    mortality_coeff_per_yr,
    assim_efficiency,
):
    """Initialize default parameters for fish"""
    global _fish_defaults

    _fish_defaults['k_metabolism'] = k_metabolism
    _fish_defaults['a_metabolism'] = a_metabolism
    _fish_defaults['b_metabolism'] = b_metabolism
    _fish_defaults['k_encounter'] = k_encounter
    _fish_defaults['a_encounter'] = a_encounter
    _fish_defaults['b_encounter'] = b_encounter
    _fish_defaults['k_consumption'] = k_consumption
    _fish_defaults['a_consumption'] = a_consumption
    _fish_defaults['b_consumption'] = b_consumption
    _fish_defaults['mortality_type'] = mortality_type
    _fish_defaults['mortality_coeff_per_yr'] = mortality_coeff_per_yr
    _fish_defaults['assim_efficiency'] = assim_efficiency


def init_zooplankton_defaults():
    """Initialize default parameters for zooplankton"""
    global _zooplankton_defaults


def init_benthic_prey_defaults(benthic_efficiency, carrying_capacity):
    """Initialize default parameters for benthic prey"""
    global _benthic_prey_defaults
    _benthic_prey_defaults['benthic_efficiency'] = benthic_efficiency
    _benthic_prey_defaults['carrying_capacity'] = carrying_capacity


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
        **kwargs,
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
        assert (0.0 <= harvest_selectivity) and (
            harvest_selectivity <= 1.0
        ), 'harvest_selectivity must be between 0. and 1.'
        assert (0.0 <= energy_frac_somatic_growth) and (
            energy_frac_somatic_growth <= 1.0
        ), 'energy_frac_somatic_growth must be between 0. and 1.'

        self.name = name
        self.functional_type_key = functional_type
        self.functional_type = functional_types[functional_type]
        self.size_class = size_class
        self.size_class_bnds_ratio = _size_class_bnds_ratio[size_class]
        self.mass = _size_class_masses[size_class]
        self.harvest_selectivity = harvest_selectivity
        self.t_frac_pelagic_static = t_frac_pelagic_static
        self.energy_frac_somatic_growth = energy_frac_somatic_growth
        self.pelagic_demersal_coupling = pelagic_demersal_coupling
        if self.pelagic_demersal_coupling:
            assert self.functional_type in [
                functional_types[t] for t in _pdc_type_keys
            ], f"pelagic-demersal coupling not defined for '{functional_type}' functional type"

        # assign defaults
        for key, default_value in _fish_defaults.items():
            assign_key = key
            assign_value = kwargs.pop(key) if key in kwargs else default_value
            if key == 'mortality_coeff_per_yr':
                assign_key = 'mortality_coeff'
                assign_value = assign_value / 365.0

            elif key == 'mortality_type':
                assert (
                    assign_value in _mortality_type_keys
                ), f'Unknown mortality type: {assign_value}'
                assign_value = mortality_types[assign_value]
            self.__dict__[assign_key] = assign_value

        if kwargs:
            raise ValueError(f'unknown parameters: {kwargs}')

        assert (0.0 <= self.assim_efficiency) and (
            self.assim_efficiency <= 1.0
        ), 'assim_efficiency must be between 0. and 1.'

        # initialize memory for result
        self.t_frac_pelagic = domain.init_array(
            name=f'{self.name}_t_frac_pelagic',
            constant=self.t_frac_pelagic_static,
        )

        self.pdc_apply_pref = self.functional_type in _pdc_apply_pref_func_types
        self.is_zooplankton = False

    def __repr__(self):
        return f'{self.name}: {self.size_class} {self.functional_type_key}'

    @property
    def is_demersal(self):
        """Return `True` if key is a demersal functional type"""
        return self.functional_type_key in _demersal_functional_type_keys

    @property
    def is_small(self):
        """Return `True` if size_class is small"""
        return self.size_class == 'small'


class zooplankton_type(object):
    """Data structure containing zooplankton parameters."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.functional_type_key = 'zooplankton'
        self.functional_type = functional_types['zooplankton']
        self.is_demersal = False
        self.is_small = False
        self.is_zooplankton = True
        for key, default_value in _zooplankton_defaults.items():
            assign_key = key
            assign_value = kwargs.pop(key) if key in kwargs else default_value
            self.__dict__[assign_key] = assign_value
        if kwargs:
            raise ValueError(f'unknown parameters: {kwargs}')


class benthic_prey_type(object):
    """Data structure containing benthic prey parameters."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.functional_type_key = 'benthic_prey'
        self.functional_type = functional_types['benthic_prey']
        self.is_zooplankton = False
        self.is_small = False

        for key, default_value in _benthic_prey_defaults.items():
            assign_key = key
            assign_value = kwargs.pop(key) if key in kwargs else default_value
            self.__dict__[assign_key] = assign_value

        if kwargs:
            raise ValueError(f'unknown parameters: {kwargs}')

        self.lcarrying_capacity = not self.carrying_capacity == 0.0

    @property
    def is_demersal(self):
        """Return `True` if key is a demersal functional type"""
        return self.functional_type_key in _demersal_functional_type_keys


class fishing(object):
    """Data structure containing fishing parameters"""

    def __init__(self, fishing_rate_per_year):
        self.fishing_rate = domain.init_array(
            name='fishing_rate',
            constant=fishing_rate_per_year / 365.0,
            attrs={'long_name': 'Imposed fishing rate', 'units': '1/d'},
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

        # index (of "from" part of link) into groups; used for biomass
        self.ndx_from = [np.where(link['from'] == all_groups)[0][0] for link in routing_settings]

        fish_names = np.array([f.name for f in fish_list], dtype=object)
        # indices of "to" and "from" part of links into fish; used from rest of terms
        # (reproduction rate, growth rate, actual recruitment rate)
        self.i_fish = [np.where(link['to'] == fish_names)[0][0] for link in routing_settings]
        self.i_fish_from = [np.where(link['from'] == fish_names)[0][0] for link in routing_settings]

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
        for i in range(self._n_links):
            yield reproduction_link(
                self.ndx_from[i],
                self.i_fish_from[i],
                self.i_fish[i],
                self.is_larval[i],
                self.efficiency[i],
            )


class reproduction_link(object):
    """Data structure with the information pertaining to a specific link in
    reproduction routing list."""

    def __init__(self, ndx_from, i_fish_from, i_fish, is_larval, efficiency):
        assert np.isscalar(ndx_from)
        assert np.isscalar(i_fish)
        assert np.isscalar(is_larval)
        assert np.isscalar(efficiency) or efficiency is None
        self.ndx_from = ndx_from
        self.i_fish_from = i_fish_from
        self.i_fish = i_fish
        self.is_larval = is_larval
        self.efficiency = efficiency


class food_web(object):
    """Data structure defining feeding relationships."""

    def __init__(self, feeding_settings, member_obj_list):

        # ensure that predator-prey entries are unique
        pred_prey = [(p['predator'], p['prey']) for p in feeding_settings]
        assert len(set(pred_prey)) == len(pred_prey), 'non-unique predator-prey relationships'

        # store functional types
        all_groups_name = [o.name for o in member_obj_list]
        all_groups_func_type = [o.functional_type for o in member_obj_list]

        # set up food web DataFrame
        self.n_links = len(feeding_settings)
        self._index = 0

        link_predator = [link['predator'] for link in feeding_settings]
        link_prey = [link['prey'] for link in feeding_settings]
        self.preference = [link['preference'] for link in feeding_settings]

        self.fish = [f for f in member_obj_list if isinstance(f, fish_type)]
        self.fish_names = [f.name for f in self.fish]
        for f in self.fish_names:
            assert (
                f in link_predator
            ), f'{f} is not listed as a predator in the food web; all fish must eat.'

        self.predator_obj = []
        self.i_fish = []
        for pred in link_predator:
            ndx_pred = all_groups_name.index(pred)
            pred_i_obj = member_obj_list[ndx_pred]
            assert isinstance(
                pred_i_obj, fish_type
            ), f'none but `fish_type` can be predators; {pred_i_obj.name} is not a fish!'
            self.predator_obj.append(pred_i_obj)
            self.i_fish.append(self.fish_names.index(pred))

        self.prey_obj = []
        self.ndx_prey = []
        for prey in link_prey:
            ndx_prey = all_groups_name.index(prey)
            prey_i_obj = member_obj_list[ndx_prey]
            self.prey_obj.append(prey_i_obj)
            self.ndx_prey.append(ndx_prey)

        # link into food_web list for each predator
        self.pred_link_ndx = {
            pred: [i for i in range(self.n_links) if link_predator[i] == pred]
            for pred in np.unique(link_predator)
        }

        # link into food_web list for each prey
        self.prey_link_ndx = {
            prey: [i for i in range(self.n_links) if link_prey[i] == prey]
            for prey in np.unique(link_prey)
        }

        pred_list_prey = {
            pred: [prey for i, prey in enumerate(link_prey) if link_predator[i] == pred]
            for pred in np.unique(link_predator)
        }
        self.pred_ndx_prey = {}
        self.pred_prey_func_type = {}
        for pred, prey_list in pred_list_prey.items():
            self.pred_ndx_prey[pred] = []
            self.pred_prey_func_type[pred] = []
            for prey in prey_list:
                i = all_groups_name.index(prey)
                self.pred_ndx_prey[pred].append(i)
                self.pred_prey_func_type[pred].append(all_groups_func_type[i])

        prey_list_pred = {
            prey: [pred for i, pred in enumerate(link_predator) if link_prey[i] == prey]
            for prey in np.unique(link_prey)
        }
        self.prey_ndx_pred = {}
        for prey, pred_list in prey_list_pred.items():
            self.prey_ndx_pred[prey] = []
            for pred in pred_list:
                i = all_groups_name.index(pred)
                self.prey_ndx_pred[prey].append(i)

        pred_prey_preference = {
            pred: {
                prey: pref
                for pref, pred_i, prey in zip(self.preference, link_predator, link_prey)
                if pred_i == pred
            }
            for pred in np.unique(link_predator)
        }
        self.pred_prey_preference = xr.DataArray(
            np.zeros((len(np.unique(link_predator)), len(np.unique(link_prey)))),
            dims=('predator', 'prey'),
            coords={'predator': np.unique(link_predator), 'prey': np.unique(link_prey)},
        )
        for i, pred in enumerate(np.unique(link_predator)):
            for j, prey in enumerate(np.unique(link_prey)):
                if prey in pred_prey_preference[pred]:
                    self.pred_prey_preference.data[i, j] = pred_prey_preference[pred][prey]

        self.zoo_names = [
            g
            for g, ft in zip(all_groups_name, all_groups_func_type)
            if ft == functional_types['zooplankton']
        ]

    def _pred_ndx_prey_filt(self, predator, prey_functional_type=None):
        """Return the index of a predator's prey in the `biomass` array;
        optionally filter by the functional type of the prey.

        Parameters
        ----------

        predator : string
          The predator whose prey index to return.

        prey_functional_type : list
          List of functional_type codes.
        """

        ndx_prey = self.pred_ndx_prey[predator]
        if prey_functional_type is not None:
            assert not set(prey_functional_type) - set(
                functional_types.values()
            ), f'unrecognized functional type requested: {prey_functional_type}'
            ndx_prey = [
                ix
                for i, ix in enumerate(ndx_prey)
                if self.pred_prey_func_type[predator][i] in prey_functional_type
            ]
        return ndx_prey

    def get_prey_biomass(
        self,
        biomass,
        predator,
        apply_preference=False,
        prey_functional_type=None,
    ):
        """Return biomass of prey.

        Parameters
        ----------

        biomass : xarray.DataArray
          Biomass array.

        predator : string
          Name of the predator.

        apply_preference : boolean, optional
          Return the prey concentration multiplied by the feeding preference.

        prey_functional_type : list, optional
          Return only prey of `functional_type in prey_functional_type`

        Returns
        -------

        prey_biomass : xr.DataArray
          The total prey biomass.

        """

        ndx_prey = self._pred_ndx_prey_filt(predator, prey_functional_type)
        biomass_prey = biomass.isel(group=ndx_prey)

        if apply_preference:
            preference = self.pred_prey_preference.sel(
                predator=predator, prey=biomass_prey.group.values
            ).rename({'prey': 'group'})
            biomass_prey *= preference

        return biomass_prey.sum('group')

    def get_consumption(self, consumption, predator=None, prey=None):
        """get the total consumption rate across all prey"""

        assert consumption.shape[0] == self.n_links, 'consumption array has wrong dims'
        assert (predator is not None) or (
            prey is not None
        ), 'arguments `predator` and `prey` cannot both be None'

        ndx_pred = []
        ndx_prey = []
        if predator is not None:
            if predator not in self.pred_link_ndx:
                return None
            ndx_pred = self.pred_link_ndx[predator]

        if prey is not None:
            if prey not in self.prey_link_ndx:
                return None
            ndx_prey = self.prey_link_ndx[prey]

        ndx = (
            [n for n in ndx_pred if n in ndx_prey] if ndx_pred and ndx_prey else ndx_pred + ndx_prey
        )

        if predator is not None and prey is None:
            return (
                consumption.isel(feeding_link=ndx)
                .reset_index(
                    ['feeding_link'],
                    drop=True,
                )
                .set_index(feeding_link='prey')
                .rename(feeding_link='group')
            )

        elif predator is None and prey is not None:
            return (
                consumption.isel(feeding_link=ndx)
                .reset_index('feeding_link', drop=True)
                .set_index(feeding_link='predator')
                .rename(feeding_link='group')
            )

        else:
            return consumption.isel(feeding_link=ndx)

    def _get_biomass_zoo_pred(self, biomass, zoo_name):
        return biomass.isel(group=self.prey_ndx_pred[zoo_name])

    def __len__(self):
        return self.n_links

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self):
            self._index = 0
            raise StopIteration
        i = self._index
        self._index += 1
        return food_web_link(self, i)


class food_web_link(object):
    """Information describing an individual feeding link."""

    def __init__(self, food_web, i):
        self.predator = food_web.predator_obj[i]
        self.prey = food_web.prey_obj[i]
        self.ndx_prey = food_web.ndx_prey[i]
        self.preference = food_web.preference[i]
        self.i_fish = food_web.i_fish[i]
