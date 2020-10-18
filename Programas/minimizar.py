# -*- coding: utf-8 -*-
# Solo enfocado a minimizar la función
#
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
#
#def f_log(x, r, u):
#    return np.log(x*(1+u) + 2*r)
#
def f_min(x):
    return np.log(x + 0.2)

agente = 10
k = 10
r = 0.1

x_op = np.arange(agente)
b1: float = 0.01
b2: float = 0.99

for i in np.arange(agente):
    print('________', i, '____________')
    x_op[i] = fmin(f_min, b1)
    print(x_op[i])

t = np.arange(agente)
plt.plot(t, x_op)
plt.xlabel('agentes')
plt.ylabel('optimos')
