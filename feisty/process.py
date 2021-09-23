import numpy as np

from . import domain


def metabolism(param, Tp, Tb, tdif, wgt):
    """
    Compute metabolic rate.
    Tp: pelagic temp
    Tb: bottom temp
    tdif: frac pelagic time
    wgt: ind weight of size class
    fcrit: feeding level to meet resting respiration rate
    cmax: max consumption rate
    U: swimming speed
    """
    temp = (Tp * tdif) + (Tb * (1.0 - tdif))

    # Own Fn ------------
    # Metabolism with its own coeff, temp-sens, mass-sens
    return (np.exp(param.kt * (temp - 10.0)) * param.amet * wgt ** (-param.bpow)) / 365.0


def encounter(param, Tp, Tb, wgt, prey, tpel, tprey, pref):
    """
    Compute encounter rates.

    Parameters
    ----------
    Tp: float
      Pelagic temperature.

    Tb: float
      Bottom temperature.

    wgt: float
      ind weight of size class.

    pred: float
       Predator biomass density.

    prey: float
      Prey biomass density.

    A: float
      Predator search rate.

    tpel: float
      Time spent in pelagic.

    tprey: float
      Time spent in area with that prey item.

    pref: float
      Preference for prey item.

    Returns
    -------
    encounter_rate: float
      The rate of prey encoutered.

    """

    temp = (Tp * tpel) + (Tb * (1.0 - tpel))

    # encounter rate
    A = (np.exp(param.ke * (temp - 10.0)) * param.gam * wgt ** (-param.benc)) / 365.0

    # Encounter per predator, mult by biomass later
    frac = np.zeros((domain.NX, 1))
    frac[tprey > 0] = 1.0

    return prey * A * frac * pref
