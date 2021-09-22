import numpy 



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
    
    temp = (Tp * tpel) + (Tb * (1. - tpel))
    
    # encounter rate
    A = (np.exp(param.ke * (temp - 10.)) * param.gam * wgt**(-param.benc)) / 365.0
    
    # Encounter per predator, mult by biomass later
    frac = np.zeros((domain.NX, 1))
    frac[tprey > 0] = 1.
    
    return prey * A * frac * pref

