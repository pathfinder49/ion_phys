"""Doppler Cooling Transient using Optical Bloch Equations"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

from ion_phys.ions.ca43 import Ca43, ground_level, P32
from ion_phys import Laser
from ion_phys.obe import OBEMatrices

ion = Ca43(B=288e-4, level_filter=[ground_level, P32])
num_states = ion.num_states
obe = OBEMatrices(ion)

idx0 = ion.index(ground_level, 4, F=4)
idx1 = ion.index(P32, 5, F=5)
delta0 = ion.delta(idx0, idx1)
I = 100
lasers = [Laser("393", q=1, I=I, delta=delta0),]

t_vec = np.linspace(0, 4/135e6, 200)

c_p0 = np.zeros(len(t_vec))
c_p1 = np.zeros(len(t_vec))

ground_sub = np.r_[ion.slice(ground_level)]
n_ground = len(ground_sub)
ground_sub = np.ix_(ground_sub, ground_sub)

init = np.zeros(obe.spont_mat.shape, dtype=np.complex128)


init[ion.index(ground_level, 4),ion.index(ground_level, 4)] = 1.
assert np.trace(init)==1.0

p_sub = np.r_[ion.slice(P32)]
p_sub = np.ix_(p_sub, p_sub)

c_p0[0] = init[idx0, idx0]
c_p1[0] = init[idx1, idx1]

convert_real = False
if not convert_real:
    int_cmat = ode(
        lambda t, cmat, lasers: obe.get_cdot(
            cmat.reshape((num_states, num_states)), t, lasers)
            .reshape(num_states * num_states))
    int_cmat.set_integrator('zvode')
    int_cmat.set_initial_value(init.flatten(), 0)
else:
    int_cmat = ode(
        lambda t, cmat, lasers: obe.get_cdot(
            cmat.view(dtype=np.complex128).reshape((num_states, num_states)),
            t, lasers).reshape(num_states * num_states).view(dtype=np.float64))
    int_cmat.set_integrator('dop853')
    int_cmat.set_initial_value(init.flatten().view(dtype=np.float64), 0)
int_cmat.set_f_params(lasers)

print("integrating cmat")
for idx, t in enumerate(t_vec[1:],1):
    cmat = int_cmat.integrate(t).view(
        dtype=np.complex128).reshape((num_states,num_states,))
    c_p0[idx] = cmat[idx0,idx0]
    c_p1[idx] = cmat[idx1,idx1]
    # print(np.trace(cmat))

fig = plt.figure("Rabi Oscillation")
plt.plot(t_vec * 132e6, c_p0, label="cmat s+4")
plt.plot(t_vec * 132e6, c_p1, label="cmat p+5")
plt.xlabel("time /(Decay time)")
plt.ylabel("population in the Stretch states states")
plt.legend()
plt.grid()
plt.show()
