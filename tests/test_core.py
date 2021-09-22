import pytest


import feisty.settings as param
import feisty.processes

@pytest.mark.parametrize("Tp", [15.])
@pytest.mark.parametrize("Tb", [3.])
@pytest.mark.parametrize("wgt", [10.])
@pytest.mark.parametrize("prey", [5.])
@pytest.mark.parametrize("tpel", [0.5])
@pytest.mark.parametrize("tprey", [0.5])
@pytest.mark.parametrize("pref", [0.9])
def test_mean_func(Tp, Tb, wgt, prey, tpel, tprey, pref):
    enc = feisty.processes.encounter(param, Tp, Tb, wgt, prey, tpel, tprey, pref)
    print(enc)
    assert isinstance(results, xr.Dataset)
