import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

import controlbenchmarks
from probsafety import utils

def test_wrapper():
    np.random.seed(42)

    batch_size = 10
    sys = controlbenchmarks.models.sys_variables["F1"]
    hit_chance = 0.7
    period = 0.02
    controller_design = 'delay_lqr'
    x0 = np.asarray([1., 1.])
    time_horizon = 2

    devs = utils.wrapper(batch_size, sys, hit_chance, period, controller_design, x0, time_horizon)

    print(devs)
    assert devs is not None
