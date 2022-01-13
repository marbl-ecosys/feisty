import numpy as np
import xarray as xr

from . import domain, ecosystem, process, settings


class feisty_instance_type(object):
    """
    This is the primary interface to the FEISTY model.

    Parameters
    ----------

    domain_dict : dict
      Dictionary containing domain information, for example::

          domain_dict = {
              'NX': len(bathymetry_data),
              'bathymetry': bathymetry_data)),
          }

    settings_dict : dict, optional
        Dictionary of model settings.

    fish_ic_data : numeric or array_like, optional
      Initial conditions for fish biomass.

    benthic_prey_ic_data : numeric or array_like
      Initial conditions for benthic prey biomass.

    Examples
    --------

    Initialize the FEISTY model and return a ``feisty_instance_type`` object::

        import feisty
        domain = get_model_domain()
        feisty_instance = feisty_instance_type(domain_dict=domain)

    In the context of time-stepping the model forward, the tendencies are computed as follows::

        dXdt = feisty_instance.compute_tendencies(...)

    Then information from the computation can be accessed as an `xarray.Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_ via the ``tendency_data`` property::

        ds_t = feisty_instance.tendency_data

    """

    def __init__(
        self,
        domain_dict,
        settings_dict={},
        fish_ic_data=None,
        benthic_prey_ic_data=None,
    ):
        """Initialize the ``feisty_instance_type``."""

        self.domain_dict = domain_dict
        self.settings_dict = settings.get_defaults()
        self.settings_dict.update(settings_dict)

        self.loffline = self.settings_dict['loffline']
        self._init_domain(self.domain_dict)

        self._init_model_settings(self.settings_dict['model_settings'])

        self._init_zooplankton(self.settings_dict['zooplankton'])
        self._init_fish_settings(self.settings_dict['fish'])
        self._init_benthic_prey_settings(self.settings_dict['benthic_prey'])

        fish_ic_data = 1e-5 if fish_ic_data is None else fish_ic_data
        benthic_prey_ic_data = 1e-5 if benthic_prey_ic_data is None else benthic_prey_ic_data
        self._init_biomass(fish_ic_data, benthic_prey_ic_data)

        self._init_food_web(self.settings_dict['food_web'])
        self._init_reproduction_routing(self.settings_dict['reproduction_routing'])
        self._init_fishing(self.settings_dict['fishing'])
        self._init_tendency_arrays()

        self.gcm_state = gcm_state_type()

    def _init_domain(self, domain_dict):
        """initialize domain"""
        domain.init_module_variables(**domain_dict)

    def _init_model_settings(self, model_settings):
        """initialize model settings"""
        ecosystem.init_module_variables(**model_settings)

    def _init_zooplankton(self, zooplankton_settings):

        ecosystem.init_zooplankton_defaults(**zooplankton_settings['defaults'])

        self.zooplankton = []
        self.zoo_names = []
        for z_settings in zooplankton_settings['members']:
            zoo_i = ecosystem.zooplankton_type(**z_settings)
            self.zooplankton.append(zoo_i)
            self.zoo_names.append(zoo_i.name)

        self.n_zoo = len(self.zooplankton)

        if self.loffline:
            self.zoo_mortality = domain.init_array_2d(
                coord_name='zooplankton',
                coord_values=self.zoo_names,
                name='zoo_mortality',
            )
        else:
            self.zoo_mortality = None

    def _init_fish_settings(self, fish_settings):
        """initialize fish"""

        ecosystem.init_fish_defaults(**fish_settings['defaults'])

        self.fish = []
        self.fish_names = []
        for fish_parameters in fish_settings['members']:
            fish_i = ecosystem.fish_type(**fish_parameters)
            self.fish.append(fish_i)
            self.fish_names.append(fish_i.name)

        assert len(set(self.fish_names)) == len(self.fish_names), 'fish are not unique'

        self.n_fish = len(self.fish)

    def _init_benthic_prey_settings(self, benthic_prey_settings):

        ecosystem.init_benthic_prey_defaults(**benthic_prey_settings['defaults'])

        self.benthic_prey = []
        self.benthic_prey_names = []
        for b_settings in benthic_prey_settings['members']:
            bprey_i = ecosystem.benthic_prey_type(**b_settings)
            self.benthic_prey.append(bprey_i)
            self.benthic_prey_names.append(bprey_i.name)

        self.n_benthic_prey = len(self.benthic_prey)

    def _init_biomass(self, fish_ic_data, benthic_prey_ic_data):
        """Initialize `biomass` array."""

        group_coord = self.zoo_names + self.fish_names + self.benthic_prey_names
        self.member_obj_list = self.zooplankton + self.fish + self.benthic_prey
        assert len(set(group_coord)) == len(
            group_coord
        ), f'duplicate names across ecosystem member groups: {group_coord}'

        n = len(group_coord)
        self.ndx_zoo = np.arange(0, self.n_zoo)
        self.ndx_fish = np.arange(self.n_zoo, self.n_zoo + self.n_fish)
        self.ndx_benthic_prey = np.arange(self.n_zoo + self.n_fish, n)

        self.ndx_prognostic = np.concatenate((self.ndx_fish, self.ndx_benthic_prey))
        self.prog_ndx_fish = np.arange(0, self.n_fish)
        self.prog_ndx_benthic_prey = np.arange(self.n_fish, self.n_fish + self.n_benthic_prey)

        # TODO: make private
        self.biomass = domain.init_array_2d('group', group_coord, name='biomass')

        self._set_fish_biomass(fish_ic_data)
        self._set_benthic_prey_biomass(benthic_prey_ic_data)

    def _init_food_web(self, feeding_settings):
        """initialize food_web"""
        self.food_web = ecosystem.food_web(
            feeding_settings,
            self.member_obj_list,
        )

    def _init_reproduction_routing(self, routing_settings):
        self.reproduction_routing = ecosystem.reproduction_routing(
            routing_settings,
            self.fish,
            self.biomass.group,
        )

    def _init_fishing(self, fishing_settings):
        self.fishing = ecosystem.fishing(**fishing_settings)

    def _init_tendency_arrays(self):
        """initialize components of the computation"""
        self.tendency_data = _init_tendency_data(
            self.zooplankton,
            self.fish,
            self.benthic_prey,
            self.food_web,
        )

    def _set_fish_biomass(self, data):
        """Set the values of the fish biomass data.

        Parameters
        ----------

        data : array_like
          Fish biomass. Must have shape = (n_fish, NX).
        """
        if np.ndim(data) != 0:
            assert data.shape == (
                self.n_fish,
                self.biomass.shape[1],
            ), 'data has the wrong dimensions'

        self.biomass.data[self.ndx_fish, :] = data

    def _set_zoo_biomass(self, data):
        """Set the values of the zooplankton biomass data.

        Parameters
        ----------

        data : array_like
          Zooplankton biomass. Must have shape = (n_zoo, NX).
        """
        if np.ndim(data) != 0:
            assert data.shape == (
                self.n_zoo,
                self.biomass.shape[1],
            ), 'data has the wrong dimensions'

        self.biomass.data[self.ndx_zoo, :] = data

    def _set_zoo_mortality(self, data):
        """Set the values of the zooplankton mortality data.

        Parameters
        ----------

        data : array_like
          Zooplankton mortality. Must have shape = (n_zoo, NX).
        """
        if np.ndim(data) != 0:
            assert data.shape == self.zoo_mortality.shape, 'data has the wrong dimensions'

        self.zoo_mortality.data[:, :] = data

    def _set_benthic_prey_biomass(self, data):
        """Set the values of the benthic prey biomass data.

        Parameters
        ----------

        data : array_like
          Benthic prey biomass. Must have shape = (n_benthic_prey, NX).
        """
        if np.ndim(data) != 0:
            assert data.shape == (
                self.n_benthic_prey,
                self.biomass.shape[1],
            ), 'data has the wrong dimensions'

        self.biomass.data[self.ndx_benthic_prey, :] = data

    def get_prognostic(self):
        """Return array of prognostic biomass components."""
        return self.biomass.isel(group=self.ndx_prognostic)

    def _compute_t_frac_pelagic(self, reset=False):
        process.compute_t_frac_pelagic(
            self.tendency_data.t_frac_pelagic,
            fish_list=self.fish,
            biomass=self.biomass,
            food_web=self.food_web,
            pelagic_functional_types=ecosystem.pelagic_functional_types,
            demersal_functional_types=ecosystem.demersal_functional_types,
            PI_be_cutoff=ecosystem.PI_be_cutoff,
            reset=reset,
        )

    def _compute_temperature(self):
        for i in range(self.n_fish):
            self.tendency_data.T_habitat[i, :] = process.t_weighted_mean_temp(
                self.gcm_state.T_pelagic,
                self.gcm_state.T_bottom,
                self.tendency_data.t_frac_pelagic[i, :],
            )

    def _update_benthic_biomass(self):
        process.compute_benthic_biomass_update(
            self.tendency_data.benthic_biomass_new,
            self.tendency_data.consumption_rate_link,
            self.benthic_prey,
            self.biomass,
            self.food_web,
            self.gcm_state.poc_flux,
        )
        self._set_benthic_prey_biomass(self.tendency_data.benthic_biomass_new)

    def _compute_pred_encounter_consumption_max(self):
        process.compute_pred_encounter_consumption_max(
            self.tendency_data.encounter_rate_pred,
            self.tendency_data.consumption_rate_max_pred,
            self.tendency_data.T_habitat,
            self.fish,
        )

    def _compute_encounter(self):
        process.compute_encounter(
            self.tendency_data.encounter_rate_link,
            self.tendency_data.encounter_rate_total,
            self.tendency_data.encounter_rate_pred,
            self.biomass,
            self.tendency_data.T_habitat,
            self.tendency_data.t_frac_pelagic,
            self.food_web,
        )

    def _compute_consumption(self):
        process.compute_consumption(
            self.tendency_data.consumption_rate_link,
            self.tendency_data.consumption_rate_max_pred,
            self.tendency_data.encounter_rate_link,
            self.tendency_data.encounter_rate_total,
            self.tendency_data.T_habitat,
            self.food_web,
        )

    def _compute_rescale_zoo_consumption(self):
        process.compute_rescale_zoo_consumption(
            self.tendency_data.consumption_rate_link,
            self.tendency_data.consumption_zoo_frac_mort,
            self.tendency_data.consumption_zoo_scaled,
            self.tendency_data.consumption_zoo_raw,
            self.biomass,
            self.zoo_mortality,
            self.food_web,
        )

    def _compute_metabolism(self):
        process.compute_metabolism(
            self.tendency_data.metabolism_rate,
            self.fish,
            self.tendency_data.T_habitat,
        )

    def _compute_ingestion(self):
        process.compute_ingestion(
            self.tendency_data.ingestion_rate,
            self.tendency_data.consumption_rate_link,
            self.food_web,
        )

    def _compute_predation(self):
        process.compute_predation(
            self.tendency_data.predation_flux,
            self.tendency_data.predation_zoo_flux,
            self.tendency_data.consumption_rate_link,
            self.biomass,
            self.food_web,
        )
        self.tendency_data.predation_rate[:, :] = (
            self.tendency_data.predation_flux.data / self.biomass.isel(group=self.ndx_fish).data
        )

    def _compute_mortality(self):
        process.compute_natural_mortality(
            self.tendency_data.mortality_rate,
            self.fish,
            self.tendency_data.T_habitat,
            ecosystem.mortality_types,
        )

    def _compute_fish_catch(self):
        process.compute_fish_catch(
            self.tendency_data.fish_catch_rate,
            self.fishing.fishing_rate,
            self.fish,
        )

    def _compute_energy_avail(self):
        process.compute_energy_avail(
            self.tendency_data.energy_avail_rate,
            self.tendency_data.ingestion_rate,
            self.tendency_data.metabolism_rate,
            self.fish,
        )

    def _compute_growth(self):
        process.compute_growth(
            self.tendency_data.growth_rate,
            self.tendency_data.energy_avail_rate,
            self.tendency_data.predation_rate,
            self.tendency_data.mortality_rate,
            self.tendency_data.fish_catch_rate,
            self.fish,
        )

    def _compute_reproduction(self):
        process.compute_reproduction(
            self.tendency_data.reproduction_rate,
            self.tendency_data.growth_rate,
            self.tendency_data.energy_avail_rate,
            fish_list=self.fish,
        )

    def _compute_recruitment(self):
        process.compute_recruitment(
            self.tendency_data.recruitment_flux,
            self.tendency_data.reproduction_rate,
            self.tendency_data.growth_rate,
            self.biomass,
            self.reproduction_routing,
        )

    def _compute_total_tendency(self):
        process.compute_total_tendency(
            self.tendency_data.total_tendency,
            self.tendency_data.recruitment_flux,
            self.tendency_data.energy_avail_rate,
            self.tendency_data.growth_rate,
            self.tendency_data.reproduction_rate,
            self.tendency_data.mortality_rate,
            self.tendency_data.predation_flux,
            self.tendency_data.fish_catch_rate,
            self.biomass,
            self.fish,
        )

    def compute_tendencies(
        self,
        fish_biomass,
        benthic_prey_biomass,
        zooplankton_biomass,
        zoo_mortality_data,
        **gcm_state_update_kwargs,
    ):
        """Compute time tendency for FEISTY model.

        Parameters
        ----------

        fish_biomass : array_like
          Array of fish biomass data.

        benthic_prey_biomass : array_like
          Array of benthic prey biomass data.

        zooplankton_biomass : array_like
          Zooplankton forcing data.

        zoo_mortality_data : array_like
          Zooplankton mortality cap.

        gcm_state_update_kwargs : dict
          Keyword, value (array) pairs with GCM state information.
        """

        # update state information
        self._set_fish_biomass(fish_biomass)
        self._set_benthic_prey_biomass(benthic_prey_biomass)
        self._set_zoo_biomass(zooplankton_biomass)
        self._set_zoo_mortality(zoo_mortality_data)
        self.gcm_state.update(**gcm_state_update_kwargs)

        # advance benthic prey concentrations
        self._update_benthic_biomass()

        # compute temperature terms
        self._compute_t_frac_pelagic()
        self._compute_temperature()

        # compute tendency components (order matters)
        self._compute_metabolism()
        self._compute_pred_encounter_consumption_max()
        self._compute_encounter()
        self._compute_consumption()
        self._compute_rescale_zoo_consumption()
        self._compute_ingestion()
        self._compute_predation()
        self._compute_mortality()

        self._compute_fish_catch()
        self._compute_energy_avail()
        self._compute_growth()
        self._compute_reproduction()
        self._compute_recruitment()

        # aggregate tendency terms
        self._compute_total_tendency()

        return self.tendency_data.total_tendency


def _init_tendency_data(zoo_list, fish_list, benthic_prey_list, food_web):
    """Return an xarray.Dataset with initialized tendency data arrays."""

    fish_names = [f.name for f in fish_list]
    zoo_names = [z.name for z in zoo_list]
    benthic_prey_names = [b.name for b in benthic_prey_list]

    pred_names = [link.predator.name for link in food_web]
    prey_names = [link.prey.name for link in food_web]
    feeding_link_coord = xr.DataArray(
        [f'{pred}_{prey}' for pred, prey in zip(pred_names, prey_names)], dims='feeding_link'
    )
    add_coords = dict(
        predator=xr.DataArray(pred_names, dims='feeding_link'),
        prey=xr.DataArray(prey_names, dims='feeding_link'),
    )
    ds = xr.Dataset(
        coords=dict(
            zooplankton=zoo_names,
            fish=fish_names,
            benthic_prey=benthic_prey_names,
            feeding_link=feeding_link_coord,
            **add_coords,
        ),
    )

    ds['t_frac_pelagic'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='t_frac_pelagic',
        attrs={
            'long_name': 'Fraction of time spent in pelagic zone',
            'units': '',
        },
    )

    for i, fish_i in enumerate(fish_list):
        ds['t_frac_pelagic'].data[i, :] = fish_i.t_frac_pelagic_static

    ds['T_habitat'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='T_habitat',
        attrs={
            'long_name': 'Temperature experienced (time-weighted mean)',
            'units': 'degC',
        },
    )
    ds['ingestion_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='ingestion_rate',
    )
    ds['predation_flux'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='predation_flux',
    )
    ds['predation_zoo_flux'] = domain.init_array_2d(
        coord_name='zooplankton',
        coord_values=zoo_names,
        name='predation_zoo_flux',
    )
    ds['predation_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='predation_rate',
    )
    ds['metabolism_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='metabolism_rate',
    )
    ds['mortality_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='mortality_rate',
    )
    ds['energy_avail_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='energy_avail',
        attrs={'long_name': 'Energy available for growth or reproduction (nu)'},
    )
    ds['growth_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='growth_rate',
        attrs={'long_name': 'Energy to somatic growth (gamma)'},
    )
    ds['reproduction_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='reproduction',
        attrs={'long_name': 'Reproduction'},
    )
    ds['recruitment_flux'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='recruitment_flux',
        attrs={'long_name': 'Recruitment from smaller size classes or reproduction'},
    )
    ds['fish_catch_rate'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='fish_catch_rate',
        attrs={'long_name': 'Specific fishing rate'},
    )
    ds['benthic_biomass_new'] = domain.init_array_2d(
        coord_name='benthic_prey',
        coord_values=benthic_prey_names,
    )

    # TODO: include benthic_prey once timestepping has been moved out of feisty
    ds['total_tendency'] = domain.init_array_2d(
        coord_name='group',
        coord_values=fish_names,
        name='total_tendency',
        attrs={'long_name': 'Total time tendency'},
    )

    ds['encounter_rate_pred'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='encounter_rate_pred',
    )
    ds['consumption_rate_max_pred'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='consumption_rate_max_pred',
    )

    ds['encounter_rate_total'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='encounter_rate_total',
    )

    ds['encounter_rate_link'] = domain.init_array_2d(
        coord_name='feeding_link',
        coord_values=feeding_link_coord,
        name='encounter_rate_link',
    ).assign_coords(add_coords)

    ds['consumption_rate_link'] = domain.init_array_2d(
        coord_name='feeding_link',
        coord_values=feeding_link_coord,
        name='consumption_rate_link',
    ).assign_coords(add_coords)

    ds['consumption_zoo_frac_mort'] = domain.init_array_2d(
        coord_name='feeding_link',
        coord_values=feeding_link_coord,
        name='consumption_zoo_frac_mort',
    )
    ds['consumption_zoo_scaled'] = domain.init_array_2d(
        coord_name='feeding_link',
        coord_values=feeding_link_coord,
        name='consumption_zoo_scaled',
    )
    ds['consumption_zoo_raw'] = domain.init_array_2d(
        coord_name='feeding_link',
        coord_values=feeding_link_coord,
        name='consumption_zoo_raw',
    )

    return ds


class gcm_state_type(object):
    """GCM state"""

    def __init__(self):
        self.T_pelagic = domain.init_array(name='T_pelagic')
        self.T_bottom = domain.init_array(name='T_bottom')
        self.poc_flux = domain.init_array(name='poc_flux')

    def update(self, T_pelagic, T_bottom, poc_flux):
        """Update the GCM state data"""
        self.T_pelagic.data[:] = T_pelagic
        self.T_bottom.data[:] = T_bottom
        self.poc_flux.data[:] = poc_flux
