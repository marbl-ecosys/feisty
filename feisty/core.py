import numpy as np
import xarray as xr

from . import domain, feeding, fish_mod, process, settings


class feisty_instance_type(object):
    """
    This is the primary interface to the FEISTY model.

    Parameters
    ----------

    domain_dict : dict
      Dictionary containing domain information, for example::

          domain_dict = {
              'NX': len(depth_of_seafloor_data),
              'depth_of_seafloor': depth_of_seafloor_data)),
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
        settings_dict=None,
        fish_ic_data=1e-5,
        benthic_prey_ic_data=1e-5,
    ):
        """Initialize the ``feisty_instance_type``."""

        self.domain_dict = domain_dict
        self.settings_dict = settings.get_defaults()
        if settings_dict is not None:
            self.settings_dict.update(settings_dict)

        self.loffline = self.settings_dict['loffline']
        self._init_domain(self.domain_dict)

        self._init_model_settings(self.settings_dict['model_settings'])

        self._init_zooplankton(self.settings_dict['zooplankton'])
        self._init_fish_settings(self.settings_dict['fish'])
        self._init_benthic_prey_settings(self.settings_dict['benthic_prey'])
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
        fish_mod.init_module_variables(**model_settings)

    def _init_zooplankton(self, zooplankton_settings):
        self.zooplankton = []
        self.zoo_names = []
        for z_settings in zooplankton_settings:
            zoo_i = fish_mod.zooplankton_type(**z_settings)
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

        self.fish = []
        self.fish_names = []
        for fish_parameters in fish_settings:
            fish_i = fish_mod.fish_type(**fish_parameters)
            self.fish.append(fish_i)
            self.fish_names.append(fish_i.name)

        assert len(set(self.fish_names)) == len(self.fish_names), 'fish are not unique'

        self.n_fish = len(self.fish)

    def _init_benthic_prey_settings(self, benthic_prey_settings):
        self.benthic_prey = []
        self.benthic_prey_names = []
        for b_settings in benthic_prey_settings:
            bprey_i = fish_mod.benthic_prey_type(**b_settings)
            self.benthic_prey.append(bprey_i)
            self.benthic_prey_names.append(bprey_i.name)

        self.n_benthic_prey = len(self.benthic_prey)

    def _init_biomass(self, fish_ic_data, benthic_prey_ic_data):
        """Initialize `biomass` array."""

        group_coord = self.zoo_names + self.fish_names + self.benthic_prey_names

        functional_type_dict = {
            o.name: o.functional_type for o in self.fish + self.benthic_prey + self.zooplankton
        }

        n = len(group_coord)
        self.ndx_zoo = np.arange(0, self.n_zoo)
        self.ndx_fish = np.arange(self.n_zoo, self.n_zoo + self.n_fish)
        self.ndx_benthic_prey = np.arange(n - 1, n)

        self.biomass = domain.init_array_2d('group', group_coord)
        self.group_func_type = xr.DataArray(
            [functional_type_dict[f] for f in group_coord],
            dims='group',
            coords={'group': self.biomass.group},
        )
        self.set_fish_biomass(fish_ic_data)
        self.set_benthic_prey_biomass(benthic_prey_ic_data)

    def _init_food_web(self, feeding_settings):
        """initialize food_web"""
        self.food_web = feeding.food_web(
            feeding_settings,
            self.fish,
            self.biomass.group,
            self.group_func_type,
        )

    def _init_reproduction_routing(self, routing_settings):
        self.reproduction_routing = fish_mod.reproduction_routing(
            routing_settings,
            self.fish,
            self.biomass.group,
        )

    def _init_fishing(self, fishing_settings):
        self.fishing = fish_mod.fishing(**fishing_settings)

    def _init_tendency_arrays(self):
        """initialize components of the computation"""
        self.tendency_data = _init_tendency_data(
            self.zooplankton,
            self.fish,
            self.benthic_prey,
        )

    def set_fish_biomass(self, data):
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

    def set_zoo_biomass(self, data):
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

    def set_zoo_mortality(self, data):
        """Set the values of the zooplankton mortality data.

        Parameters
        ----------

        data : array_like
          Zooplankton mortality. Must have shape = (n_zoo, NX).
        """
        if np.ndim(data) != 0:
            assert data.shape == self.zoo_mortality.shape, 'data has the wrong dimensions'

        self.zoo_mortality.data[:, :] = data

    def set_benthic_prey_biomass(self, data):
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

    def _compute_t_frac_pelagic(self, reset=False):
        process.compute_t_frac_pelagic(
            self.tendency_data.t_frac_pelagic,
            fish_list=self.fish,
            biomass=self.biomass,
            food_web=self.food_web,
            reset=reset,
        )

    def _compute_temperature(self):
        for i in range(self.n_fish):
            self.tendency_data.T_habitat[i, :] = process.t_weighted_mean_temp(
                self.gcm_state.T_pelagic,
                self.gcm_state.T_bottom,
                self.tendency_data.t_frac_pelagic[i, :],
            )

    def _compute_feeding(self):
        self.food_web.compute(
            biomass=self.biomass,
            T_habitat=self.tendency_data.T_habitat,
            t_frac_pelagic=self.tendency_data.t_frac_pelagic,
            zoo_mortality=self.zoo_mortality,
        )

    def _update_benthic_biomass(self):
        process.compute_benthic_biomass_update(
            self.tendency_data.benthic_biomass_new,
            benthic_prey_list=self.benthic_prey,
            biomass=self.biomass,
            food_web=self.food_web,
            poc_flux=self.gcm_state.poc_flux,
        )
        self.set_benthic_prey_biomass(self.tendency_data.benthic_biomass_new)

    def _compute_metabolism(self):
        process.compute_metabolism(
            self.tendency_data.metabolism_rate,
            self.fish,
            self.tendency_data.T_habitat,
        )

    def _compute_ingestion(self):
        process.compute_ingestion(
            self.tendency_data.ingestion_rate,
            self.food_web,
        )

    def _compute_predation(self):
        process.compute_predation(
            self.tendency_data.predation_flux,
            self.food_web,
            self.biomass,
        )
        self.tendency_data.predation_rate[:, :] = (
            self.tendency_data.predation_flux.data / self.biomass.isel(group=self.ndx_fish).data
        )

    def _compute_mortality(self):
        process.natural_mortality(
            self.tendency_data.mortality_rate,
            self.fish,
            self.tendency_data.T_habitat,
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
        self.set_fish_biomass(fish_biomass)
        self.set_benthic_prey_biomass(benthic_prey_biomass)
        self.set_zoo_biomass(zooplankton_biomass)
        self.set_zoo_mortality(zoo_mortality_data)
        self.gcm_state.update(**gcm_state_update_kwargs)

        # advance benthic prey concentrations
        self._update_benthic_biomass()

        # compute temperature terms
        self._compute_t_frac_pelagic()
        self._compute_temperature()

        # compute tendency components (order matters)
        self._compute_metabolism()
        self._compute_feeding()
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


def _init_tendency_data(zoo_list, fish_list, benthic_prey_list):
    """Return an xarray.Dataset with initialized tendency data arrays."""
    fish_names = [f.name for f in fish_list]
    zoo_names = [z.name for z in zoo_list]
    benthic_prey_names = [b.name for b in benthic_prey_list]

    ds = xr.Dataset(
        coords=dict(
            zooplankton=zoo_names,
            fish=fish_names,
            benthic_prey=benthic_prey_names,
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
    ds['total_tendency'] = domain.init_array_2d(
        coord_name='fish',
        coord_values=fish_names,
        name='total_tendency',
        attrs={'long_name': 'Total time tendency'},
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
