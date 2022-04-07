import cftime
import numpy as np
import xarray as xr

from ..core import settings


def annual_harmonic(t, mu, a1, phi1=0.0, a2=0.0, phi2=0.0):
    """A harmonic function"""
    return (
        mu
        + a1 * np.cos(2.0 * np.pi * t / 365.0 + phi1)
        + a2 * np.cos(4.0 * np.pi * t / 365.0 + phi2)
    )


def gen_idealized_cycle(name, nt, mu, a1, phi1=0.0, a2=0.0, phi2=0.0):
    """Generate a xarray.DataArray from a harmonic function."""
    time = xr.cftime_range(start=cftime.DatetimeNoLeap(1, 1, 1), periods=nt)
    return xr.DataArray(
        annual_harmonic(np.arange(nt), mu, a1, phi1, a2, phi2),
        dims=('time',),
        name=name,
        coords={'time': time},
    )


def martin_curve(ocean_depth, poc_flux_bottom_100, b):
    """The Martin curve power for export particulate organic carbon (POC)."""
    return poc_flux_bottom_100 * (ocean_depth / 100.0) ** -b


def gen_domain_dict(x, bathymetry):
    return dict(
        bathymetry=xr.DataArray(
            bathymetry,
            dims=('X'),
            name='bathymetry',
            attrs={'long_name': 'depth', 'units': 'm'},
            coords={'X': x},
        ),
        NX=len(bathymetry),
    )


def domain_tanh_shelf(nx=22):
    """Compute idealized bathymetry analytically using ``tanh`` function."""
    x = np.linspace(-0.5, 5.0, nx)
    return gen_domain_dict(x, bathymetry=30 + 10 ** (3 * np.tanh(x)))


def cycle_poc_flux_bottom(nt, domain_dict, mu=10.0e-3, amp_fraction=0.2, b=0.7):
    """Generate a seasonally-varying POC flux to the bottom over domain_dict"""

    poc_flux_bottom_100_t = gen_idealized_cycle('poc_flux_bottom', nt, mu, mu * amp_fraction)
    poc_flux_bottom = martin_curve(domain_dict['bathymetry'], poc_flux_bottom_100_t, b)
    poc_flux_bottom.attrs = {
        'long_name': 'POC flux',
        'units': 'g/m^2/d',
        'b': b,
    }
    return poc_flux_bottom


def cycle_T_pelagic(nt, domain_dict, mu=14.0, amp_fraction=0.3):
    """Generate a seasonally-varying pelagic temperature over domain_dict"""
    ones = xr.full_like(domain_dict['bathymetry'].X, fill_value=1.0)
    T_pelagic = gen_idealized_cycle('T_pelagic', nt, mu, mu * amp_fraction) * ones
    T_pelagic.attrs = {'long_name': 'T_pelagic', 'units': 'degC'}
    return T_pelagic


def cycle_T_bottom(nt, domain_dict, mu=4.0, amp_fraction=0.05):
    """Generate a seasonally-varying bottom temperature over domain_dict"""
    ones = xr.full_like(domain_dict['bathymetry'].X, fill_value=1.0)
    T_bottom = gen_idealized_cycle('T_bottom', nt, mu, mu * amp_fraction) * ones
    T_bottom.attrs = {'long_name': 'T_bottom', 'units': 'degC'}
    return T_bottom


def cycle_zooplankton(nt, domain_dict, zoo_spec=None):
    """Generate a seasonally-varying zooplankton concentration over domain_dict"""

    if zoo_spec is None:
        sd = settings.get_defaults()
        zoo_names = [z['name'] for z in sd['zooplankton']['members']]
        zoo_spec = {}
    elif isinstance(zoo_spec, list):
        zoo_names = zoo_spec
        zoo_spec = {}
    else:
        assert isinstance(zoo_spec, dict)

    if not zoo_spec:
        mu = 4.0
        amp_fraction = 0.2
        phase = 10.0
        zoo_spec = {}
        for z in zoo_names:
            zoo_spec[z] = dict(mu=mu, amp_fraction=amp_fraction, phase=phase)
            mu *= 0.5
            amp_fraction = 0.2
            phase += 30.0

    ones = xr.full_like(domain_dict['bathymetry'].X, fill_value=1.0)

    da_list = []
    parms_list = []
    for z, parms in zoo_spec.items():
        da = (
            gen_idealized_cycle(
                z, nt, parms['mu'], parms['mu'] * parms['amp_fraction'], parms['phase']
            )
            * ones
        )
        da.attrs = {'long_name': 'Zooplankton biomass', 'units': 'g/m^2'}
        parms_list.append(f'{z} = {parms}')
        da_list.append(da)

    da = xr.concat(
        da_list,
        dim=xr.DataArray(
            zoo_names,
            dims=('zooplankton'),
            name='zooplankton',
        ),
    )
    da.attrs.update({'harmonic_parms': '; '.join(parms_list)})
    return da


def zooplankton_mortality(zooC, mortality_coeff=0.07):
    da = mortality_coeff * zooC**2
    da.attrs = {'long_name': 'Zooplankton quadratic mortality', 'units': 'g/m^2/d'}
    return da


def forcing_cyclic(domain_dict, nt=365, zoo_spec=None):
    """Return forcing data for a test case using harmonic functions to generate seasonally-varying temperature, POC flux and zooplankton biomass forcing.

    Parameters
    ----------

    domain_dict : dict
      Dictionary containing ``feisty.domain`` settings.

    nt : integer
      Number of time sets (days)

    zoo_spec : dict or list
      If ``list``, specifies ``zoo_names`` (i.e., ``zoo_spec = ["zoo1", "zoo2"]``) and parameters are generated using default values. If ``dict`` then specifies a dictionary of parameters for cyclic harmonic function (``mu``, ``amp_fraction``, ``phase``), for example::

          zoo_spec = {"zoo1": "mu": 4.0, "amp_fraction": 0.2, "phase": 0.0}

    Returns
    -------

    forcing : xarray.Dataset
        The dataset with forcing variable.  For example::

            <xarray.Dataset>
            Dimensions:          (time: 365, X: 22, zooplankton: 1)
            Coordinates:
              * time             (time) float64 0.0 1.0 2.0 3.0 ... 361.0 362.0 363.0 364.0
              * X                (X) float64 -0.5 -0.2381 0.02381 0.2857 ... 4.476 4.738 5.0
              * zooplankton      (zooplankton) <U3 'Zoo'
            Data variables:
                T_pelagic        (time, X) float64 18.2 18.2 18.2 18.2 ... 18.2 18.2 18.2
                T_bottom         (time, X) float64 4.2 4.2 4.2 4.2 4.2 ... 4.2 4.2 4.2 4.2
                poc_flux_bottom  (time, X) float64 0.02785 0.02775 ... 0.002347 0.002346
                zooC             (zooplankton, time, X) float64 3.329 3.329 ... 3.321 3.321
                zoo_mort         (zooplankton, time, X) float64 0.7756 0.7756 ... 0.7722
            Attributes:
                note:     Idealized cyclic forcing for FEISTY model.
    """

    zooC = cycle_zooplankton(nt, domain_dict, zoo_spec)
    poc_flux_bottom = cycle_poc_flux_bottom(nt, domain_dict)

    assert (zooC > 0.0).all()
    assert (poc_flux_bottom >= 0.0).all()

    ds = xr.Dataset(
        dict(
            T_pelagic=cycle_T_pelagic(nt, domain_dict),
            T_bottom=cycle_T_bottom(nt, domain_dict),
            poc_flux_bottom=poc_flux_bottom,
            zooC=zooC,
            zoo_mort=zooplankton_mortality(zooC),
        )
    )
    ds.attrs['note'] = 'Idealized cyclic forcing for FEISTY model.'
    return ds
