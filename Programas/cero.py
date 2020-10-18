import numpy as np


def f_cero(v):
    for i in np.arange(len(v)):
        if v[i] < 0:
            v[i] = 0
    return v


t = np.random.rand(10)
t = -1*t
t[8] = 8
print(t)
f_cero(t)
print(t)
