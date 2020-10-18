# -*- coding: utf-8 -*-
"""
Editor de PyCharm

Este es un archivo temporal.
"""
#   A microscopic model of the stock market: cycles, booms, and crashes
#   Levy, Moshe and Levy, Haim and Solomon, Sorin
#   Economics Letters
#   1994
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def f_lls(x, wh, H):
    y = np.log((1-x)*wh*1.01+x*wh*(1+H))
    return y

tiempo = 9
agentes = 2
k = 5
Nt = 10000
d = 0.026
r = 0.01
h = 0

p = np.zeros(tiempo)
ph = np.zeros((agentes, tiempo))
Nh = np.zeros((agentes, tiempo))
wh = np.zeros((agentes, tiempo))
w = np.zeros((agentes, tiempo))
N = np.zeros((agentes, tiempo))
x_op = np.zeros((agentes, tiempo))
x_op_k = np.zeros((k, agentes, tiempo))
x = np.zeros((agentes, tiempo))
H = np.zeros((agentes, k))

w[:, 0] = 1000
p[0] = 4.4
for i in np.arange(agentes):
    H[i, :] = np.random.rand(k)
N[:, 0] = Nt/agentes

b1 = 0.00001
b2 = 0.99999
bds = [b1, b2]

for t in np.arange(tiempo):
    if t > 0:
        for i in np.arange(agentes):
            ph[i, t] = np.random.choice(H[i, :])
        for i in np.arange(agentes):
            wh[i] = w[i, t-1] + N[i, t-1]*(ph[i, t]-p[t-1])
        for i in np.arange(agentes):
            for j in np.arange(k):
                x_op_k[i, t, j] = optimize.fmin(f_lls, 0, args=(wh, H[i, j]))
            x_op[i, t] = np.sum(x_op_k[i, t, :])/k
        for i in np.arange(agentes):
            x[i, t] = np.random.rand(0.0001, 0.24)
        for i in np.arange(agentes):
            Nh[i] = (x[i, t]*wh[i])/(ph[i, t])
        p[t] = (np.sum(x[:, t]*wh[:]))/Nt
        for i in np.arange(agentes):
            w[i, t] = w[i, t-1] + N[i, t-1]*(p[t]-p[t-1])
        for i in np.arange(agentes):
            N[i, t] = (x[i, t]*w[i, t])/(p[t])
        for i in np.arange(agentes):
            for j in np.arange(k):
                if j < k-1:
                    H[j] = H[j+1]
            H[i, k-1] = (p[t]-p[t-1]+d)/p[t-1]
        for i in np.arange(agentes):
            w[i, t] = (1-x[i, t])*w[i, t]*(1+r)+x[i, t]*w[i, t]*(1+H[i, k])