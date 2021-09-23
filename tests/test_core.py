import numpy as np
import pytest

import feisty.process
import feisty.settings as param


@pytest.mark.parametrize('Tp', [15.0])
@pytest.mark.parametrize('Tb', [3.0])
@pytest.mark.parametrize('wgt', [10.0])
@pytest.mark.parametrize('prey', [5.0])
@pytest.mark.parametrize('tpel', [0.5])
@pytest.mark.parametrize('tprey', [0.5])
@pytest.mark.parametrize('pref', [0.9])
def test_mean_func(Tp, Tb, wgt, prey, tpel, tprey, pref):
    enc = feisty.process.encounter(param, Tp, Tb, wgt, prey, tpel, tprey, pref)
    np.testing.assert_almost_equal(enc, 0.51127804)
