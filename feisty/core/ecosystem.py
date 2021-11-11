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


class food_web(object):
    """define feeding relationships"""

    def __init__(self, feeding_settings, fish_list, all_groups, all_groups_func_type):

        if isinstance(all_groups, xr.DataArray):
            all_groups = all_groups.values
        elif isinstance(all_groups, list):
            all_groups = np.array(all_groups, dtype=object)

        if isinstance(all_groups_func_type, xr.DataArray):
            all_groups_func_type = all_groups_func_type.values
        elif isinstance(all_groups_func_type, list):
            all_groups_func_type = np.array(all_groups_func_type, dtype=object)

        # ensure that predator-prey entries are unique
        pred_prey = [(p['predator'], p['prey']) for p in feeding_settings]
        assert len(set(pred_prey)) == len(pred_prey), 'non-unique predator-prey relationships'

        # store functional types
        # set up food web DataFrame
        masses = {f.name: f.mass for f in fish_list}

        self.n_links = len(feeding_settings)
        self.link_predator = [link['predator'] for link in feeding_settings]
        self.link_prey = [link['prey'] for link in feeding_settings]
        preference = [link['encounter_parameters']['preference'] for link in feeding_settings]

        self.fish_names = [f.name for f in fish_list]

        for f in self.fish_names:
            assert (
                f in self.link_predator
            ), f'{f} is not listed as a predator in the food web; all fish must eat.'

        # link into food_web list for each predator
        self.pred_link_ndx = {
            pred: [i for i in range(self.n_links) if self.link_predator[i] == pred]
            for pred in np.unique(self.link_predator)
        }

        # link into food_web list for each prey
        self.prey_link_ndx = {
            prey: [i for i in range(self.n_links) if self.link_prey[i] == prey]
            for prey in np.unique(self.link_prey)
        }

        pred_list_prey = {
            pred: [prey for i, prey in enumerate(self.link_prey) if self.link_predator[i] == pred]
            for pred in np.unique(self.link_predator)
        }

        self.pred_ndx_prey = {}
        self.pred_prey_func_type = {}
        for pred, prey_list in pred_list_prey.items():
            ndx = []
            for prey in prey_list:
                assert len(np.where(all_groups == prey)[0]) == 1
                ndx.append(np.where(all_groups == prey)[0][0])
            self.pred_ndx_prey[pred] = ndx
            self.pred_prey_func_type[pred] = all_groups_func_type[ndx]

        prey_list_pred = {
            prey: [pred for i, pred in enumerate(self.link_predator) if self.link_prey[i] == prey]
            for prey in np.unique(self.link_prey)
        }

        self.prey_ndx_pred = {}
        for prey, pred_list in prey_list_pred.items():
            ndx = []
            for pred in pred_list:
                assert len(np.where(all_groups == pred)[0]) == 1
                ndx.append(np.where(all_groups == pred)[0][0])
            self.prey_ndx_pred[prey] = ndx

        pred_prey_preference = {
            pred: {
                prey: pref
                for pref, pred_i, prey in zip(preference, self.link_predator, self.link_prey)
                if pred_i == pred
            }
            for pred in np.unique(self.link_predator)
        }

        self.pred_prey_preference = xr.DataArray(
            np.zeros((len(np.unique(self.link_predator)), len(np.unique(self.link_prey)))),
            dims=('predator', 'prey'),
            coords={'predator': np.unique(self.link_predator), 'prey': np.unique(self.link_prey)},
        )
        for i, pred in enumerate(np.unique(self.link_predator)):
            for j, prey in enumerate(np.unique(self.link_prey)):
                if prey in pred_prey_preference[pred]:
                    self.pred_prey_preference.data[i, j] = pred_prey_preference[pred][prey]

        self.encounter_obj = [
            encounter_type(
                predator=link['predator'],
                prey=link['prey'],
                predator_size_class_mass=masses[link['predator']],
                **link['encounter_parameters'],
            )
            for link in feeding_settings
        ]
        self.consumption_obj = [
            consumption_type(
                predator=link['predator'],
                prey=link['prey'],
                predator_size_class_mass=masses[link['predator']],
                **link['consumption_parameters'],
            )
            for link in feeding_settings
        ]

        self.feeding_link_coord = xr.DataArray(
            [f"{link['predator']}_{link['prey']}" for link in feeding_settings], dims='feeding_link'
        )

        add_coords = dict(
            predator=xr.DataArray(self.link_predator, dims='feeding_link'),
            prey=xr.DataArray(self.link_prey, dims='feeding_link'),
        )
        self.encounter = domain.init_array_2d(
            coord_name='feeding_link',
            coord_values=self.feeding_link_coord,
            name='encouter_rate',
        ).assign_coords(add_coords)

        self.consumption = domain.init_array_2d(
            coord_name='feeding_link',
            coord_values=self.feeding_link_coord,
            name='consumption_rate',
        ).assign_coords(add_coords)

        self.consumption_max = domain.init_array_2d(
            coord_name='feeding_link',
            coord_values=self.feeding_link_coord,
            name='consumption_max',
        ).assign_coords(add_coords)

        # index into food_web links list for zooplankton prey
        self.zoo_names = [
            g
            for g, ft in zip(all_groups, all_groups_func_type)
            if ft == functional_types['zooplankton']
        ]

        self.consumption_zoo_frac_mort = {}
        self.consumption_zoo_scaled = {}
        self.consumption_zoo_raw = {}
        for zoo_i in self.zoo_names:
            feeding_link_coord_zoo_i = self.feeding_link_coord.isel(
                feeding_link=self.prey_link_ndx[zoo_i]
            )
            self.consumption_zoo_frac_mort[zoo_i] = domain.init_array_2d(
                coord_name='feeding_link_zoo',
                coord_values=feeding_link_coord_zoo_i,
                name='consumption_zoo_frac_mort',
            )
            self.consumption_zoo_scaled[zoo_i] = domain.init_array_2d(
                coord_name='feeding_link_zoo',
                coord_values=feeding_link_coord_zoo_i,
                name='consumption_zoo_scaled',
            )
            self.consumption_zoo_raw[zoo_i] = domain.init_array_2d(
                coord_name='feeding_link_zoo',
                coord_values=feeding_link_coord_zoo_i,
                name='consumption_zoo_raw',
            )

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

    def _compute_encounter(self, biomass, T_habitat, t_frac_pelagic):
        """compute encounter rate"""

        i = 0
        for pred, prey, obj in zip(self.link_predator, self.link_prey, self.encounter_obj):

            biomass_prey = biomass.sel(group=prey)
            t_frac_pelagic_pred = t_frac_pelagic.sel(fish=pred)
            t_frac_prey_pred = t_frac_pelagic_pred
            if is_demersal(prey):
                t_frac_prey_pred = 1.0 - t_frac_pelagic_pred

            obj.compute(
                self.encounter[i, :],
                biomass_prey,
                T_habitat.sel(fish=pred),
                t_frac_prey_pred,
            )
            i += 1

    def _compute_consumption(self, T_habitat):
        """compute consumption rate"""

        zipped_iterator = zip(self.link_predator, self.link_prey, self.consumption_obj)
        for i, (pred, prey, obj) in enumerate(zipped_iterator):
            obj.compute(
                self.consumption_max[i, :],
                self.consumption[i, :],
                self.encounter[i, :],
                self._get_total_encounter(pred),
                T_habitat.sel(fish=pred),
            )

    def compute(self, biomass, T_habitat, t_frac_pelagic, zoo_mortality):
        """Compute feeding rates.

        Parameters
        ----------

        biomass : xarray.DataArray
          Biomass concentration for all groups (i.e., zooplankton, fish, benthic prey).

        T_habitat : xarray.DataArray
          The average temperature experienced by fish.

        t_frac_pelagic : xarray.DataArray
          Fraction of time spent in pelagic zone.

        zoo_mortality : xarray.DataArray
          Maximum consumption for each zooplankton group.
        """

        self._compute_encounter(biomass, T_habitat, t_frac_pelagic)
        self._compute_consumption(T_habitat)
        if zoo_mortality is not None:
            self._rescale_consumption(biomass, zoo_mortality)

    def _get_total_encounter(self, predator):
        """get the total encouter rate across all prey"""
        return self.encounter.isel(feeding_link=self.pred_link_ndx[predator]).sum('feeding_link')

    def get_consumption(self, predator=None, prey=None):
        """get the total consumption rate across all prey"""
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
                self.consumption.isel(feeding_link=ndx)
                .reset_index(
                    ['feeding_link'],
                    drop=True,
                )
                .set_index(feeding_link='prey')
                .rename(feeding_link='group')
            )

        elif predator is None and prey is not None:
            return (
                self.consumption.isel(feeding_link=ndx)
                .reset_index('feeding_link', drop=True)
                .set_index(feeding_link='predator')
                .rename(feeding_link='group')
            )

        else:
            return self.consumption.isel(feeding_link=ndx)

    def _get_biomass_zoo_pred(self, biomass, zoo_name):
        return biomass.isel(group=self.prey_ndx_pred[zoo_name])

    def _rescale_consumption(self, biomass, zoo_mortality):
        """limit zooplankton consumption by mortality term"""

        for zoo_i in self.zoo_names:
            biomass_zoo_pred = self._get_biomass_zoo_pred(biomass, zoo_i)

            bio_con_zoo = biomass_zoo_pred * self.get_consumption(prey=zoo_i)
            bio_con_zoo_sum = bio_con_zoo.sum('group')

            zoo_mortality_i = zoo_mortality.sel(zooplankton=zoo_i)

            self.consumption_zoo_frac_mort[zoo_i].data[:, :] = bio_con_zoo_sum / (
                zoo_mortality_i + constants.eps
            )

            bio_con_zoo_scaled = (bio_con_zoo / bio_con_zoo_sum) * zoo_mortality_i

            self.consumption_zoo_scaled[zoo_i].data[:, :] = np.where(
                bio_con_zoo_sum > zoo_mortality_i,
                bio_con_zoo_scaled / biomass_zoo_pred,
                self.consumption.isel(feeding_link=self.prey_link_ndx[zoo_i]),
            )
            self.consumption_zoo_raw[zoo_i].data[:, :] = self.consumption.data[
                self.prey_link_ndx[zoo_i], :
            ]
            self.consumption.data[self.prey_link_ndx[zoo_i], :] = self.consumption_zoo_scaled[
                zoo_i
            ].data


class encounter_type(object):
    """Data structure to support computation of encounter rates.

    Parameters
    ----------
    predator : string
      Name of the predator.

    prey : string
      Name of the prey.

    predator_size_class_mass : float
      Mass of predator size class.

    preference : float
      Preference for prey item.

    ke : float, optional
      ke parameter [More detail here]

    gam : float, optional
      gam parameter [More detail here]

    benc : float, optional
      benc parameter [More detail here]
    """

    def __init__(
        self, predator, prey, predator_size_class_mass, preference, ke=0.063, gam=70.0, benc=0.20
    ):
        self.predator = predator
        self.prey = prey
        self.ke = ke
        self.gam = gam
        self.benc = benc

        self.predator_size_class_mass = predator_size_class_mass
        self.preference = preference

    def __repr__(self):
        return f'enc_{self.predator}_{self.prey}'

    def compute(self, da, biomass_prey, T_habitat, t_frac_prey):
        """
        Compute encounter rates.

        Parameters
        ----------
        da : xarray.DataArray
          DataArray to be filled

        biomass_prey : float
          Prey biomass density.

        T_habitat : array_like
           Experienced temperature.

        t_frac_pelagic : float
          Fraction of time spent in pelagic.

        t_frac_prey : float
          Time spent in area with that prey item.
        """
        if self.preference == 0:
            return

        # encounter rate
        A = (
            np.exp(self.ke * (T_habitat - 10.0))
            * self.gam
            * self.predator_size_class_mass ** (-self.benc)
        ) / 365.0

        da[:] = xr.where(
            t_frac_prey > 0,
            biomass_prey * A * self.preference,
            0.0,
        )


class consumption_type(object):
    def __init__(
        self,
        predator,
        prey,
        predator_size_class_mass,
        kc=0.063,
        h=20.0,
        bcmx=0.25,
    ):

        self.predator = predator
        self.prey = prey

        self.kc = kc
        self.h = h
        self.bcmx = bcmx

        self.predator_size_class_mass = predator_size_class_mass

        self.result = domain.init_array(name=self.__repr__)

    def __repr__(self):
        return f'con_{self.predator}_{self.prey}'

    def compute(self, consumption_max, consumption, encounter, encounter_total, T_habitat):
        """
        Tp: pelagic temp
        Tb: bottom temp
        tpel: frac pelagic time
        wgt: ind weight of size class
        enc: array of all encountered food
        calculates consumption rate
        """

        # Cmax rate
        consumption_max[:] = (
            np.exp(self.kc * (T_habitat - 10.0))
            * self.h
            * self.predator_size_class_mass ** (-self.bcmx)
        ) / 365.0

        consumption[:] = (
            consumption_max[:] * encounter[:] / (consumption_max[:] + encounter_total[:])
        )
