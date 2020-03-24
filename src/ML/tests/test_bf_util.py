import pytest
import random
from bf_util import numpy_util, trig_util

def test_running_stats_raises_error():
    with pytest.raises(ValueError):
        numpy_util.running_mean(None,1,5)

def test_add_polar():
    d = {"RE": [[random.random()]*5], "IM":[[random.random()]*5]}
    results = trig_util.add_polar(d)
    for k in ["amp", "phase"]:
        assert k in results
        assert len(results[k]) == 1
        assert len(results[k][0]) == 5
