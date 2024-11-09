import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

import controlbenchmarks
import probsafety

# Testing nominal trajectory
sys = controlbenchmarks.models.sys_variables["F1"]
period = 0.02
time_horizon = 2

Q, R = np.eye(sys.nstates), np.eye(sys.ninputs)
K, _, _ = ctrl.lqr(sys, Q, R)
u = lambda x, t: -K @ x
xsn = probsafety.sim.nominal_trajectory(sys, period, 2, [1, 1], u)

assert np.isclose(xsn[-1,:], np.zeros(sys.nstates),  atol=1e-5).all(), "Nominal trajectory fails to converge to the origin."

# plt.plot(xsn)
# plt.show()

# Testing dsim with zero delay
dsys = ctrl.c2d(sys, period)
K, _, _ = ctrl.dlqr(dsys, Q, R)
u = lambda x, t: -K @ x

xs = probsafety.sim.dsim(dsys, [1, 1], u, np.ones((100,)))

print(probsafety.sim.maxdev(xs, xsn))
assert np.isclose(xs[5:, :], xsn[5:, :], atol=1e-2).all(), "Discretized sim with no deadline misses fails to closely mimic the nominal trajectory."

# plt.plot(xs)
# plt.show()

# Testing dsim with one-period delay
asys = controlbenchmarks.controllers.augment(dsys)
K = controlbenchmarks.controllers.delay_lqr(asys, Q, R)
u = lambda x, t: -K @ x

xsd = probsafety.sim.dsim(asys, [1, 1, 0], u, np.ones((100,)))

print(probsafety.sim.maxdev(xs, xsd[:, :-1]))
print(probsafety.sim.maxdev(xsn, xsd[:, :-1]))

# plt.plot(xsd[:, :-1])
# plt.show()
