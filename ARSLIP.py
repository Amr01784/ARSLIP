import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def dfdt(x, t):
    f = np.zeros(4)
    m = 75
    ks = 25000
    ka = 2500
    Rn = 1.01
    g = 9.8
    
    f[0] = x[1]
    f[1] = x[0] * (x[3]**2) - g * np.cos(x[2]) + (ks * (Rn - x[0])) / m
    f[2] = x[3]
    f[3] = (g / x[0]) * np.sin(x[2]) - (ka * x[2]) / (m * x[0]**2) - (2 * x[1] * x[3]) / x[0]
    
    return f

tspan = np.arange(0, 2.81, 0.01)
x0 = np.array([0.965, 0, 0, -1.09])
x = odeint(dfdt, x0, tspan)

plt.plot(tspan, x[:, 0])
plt.xlabel('t')
plt.ylabel('r(t)')
plt.show()

ks = 25000
DL = np.abs(0.96151 - x[:, 0])

Fy = ks * DL * np.cos(x[:, 2]) / (9.8 * 74)
Fx = ks * DL * np.sin(x[:, 2]) / (9.8 * 74)

plt.plot(Fy)
plt.plot(Fx)
plt.show()