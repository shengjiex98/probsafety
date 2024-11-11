import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

import controlbenchmarks
from probsafety import sim

sys = controlbenchmarks.models.sys_variables["F1"]
period = 0.02
time_horizon = 2
x0 = np.asarray([1., 1.])
u = lambda x, t: -ctrl.lqr(sys, np.eye(sys.nstates), np.eye(sys.ninputs))[0] @ x

def test_generate_actions():
    n = 1000
    p = 0.5
    actions = sim.generate_actions(n, p)
    assert np.isclose(np.mean(actions), p, atol=0.1)

def test_generate_action_matrix():
    n = 1000
    p = 0.5
    batch_size = 10
    actions = sim.generate_action_matrix(n, p, batch_size)
    assert np.isclose(np.mean(actions, axis=1), p, atol=0.1).all()

def test_nominal_trajectory():
    xsn = sim.nominal_trajectory(sys, period, time_horizon, x0, u)
    # plt.plot(xsn)
    # plt.show()
    assert np.isclose(xsn[-1,:], np.zeros(sys.nstates),  atol=1e-5).all()

def test_dsim_all_hits():
    dsys = ctrl.c2d(sys, period)
    xs = sim.dsim(dsys, [1, 1], u, np.ones((100,)))
    # plt.plot(xs)
    # plt.show()
    assert np.isclose(xs[-1,:], np.zeros(sys.nstates),  atol=1e-5).all()

def test_dsim_all_misses():
    dsys = ctrl.c2d(sys, period)
    xs = sim.dsim(dsys, [1, 1], u, np.zeros((100,)))
    # plt.plot(xs)
    # plt.show()
    assert np.isclose(xs[-1,:], np.asarray([14.]),  atol=1e-5).all()

def test_maxdev():
    t1 = np.random.random((100, 2))
    t2 = np.random.random((100, 2))
    assert sim.maxdev(t1, t2) == np.max(np.sqrt(np.sum((t1 - t2)**2, axis=1)))

def test_maxdev_C():
    t1 = np.random.random((100, 2))
    t2 = np.random.random((100, 2))
    C = np.asarray([[1, 0]])
    assert sim.maxdev(t1, t2, C) == np.max(np.abs(t1[:, 0] - t2[:, 0]))

# Testing dsim with one-period delay
# asys = controlbenchmarks.controllers.augment(dsys)
# K = controlbenchmarks.controllers.delay_lqr(asys, Q, R)
# u = lambda x, t: -K @ x

# xsd = sim.dsim(asys, [1, 1, 0], u, np.ones((100,)))

# print(sim.maxdev(xs, xsd[:, :-1]))
# print(sim.maxdev(xsn, xsd[:, :-1]))

# plt.plot(xsd[:, :-1])
# plt.show()
