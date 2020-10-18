# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:35:38 2019

@author: Mauricio
"""
# Agent-based models of financial markets
# Samanidou, E. and Zschischang, E. and Stauffer, D. and Lux, T.
# 2007

import numpy as np

agentes = 10
tiempo = 10
Na = 100
r = 1

w = np.zeros((agentes,tiempo))
x = np.zeros((agentes,tiempo))
D = np. zeros((agentes,tiempo))
p = np.zeros(tiempo)

k = np.random.
H = np.zeros(k)
 
w[:,0] = 100

for t in np.arange(tiempo):
    if t >0:
        for i in np.arange(agentes):
            w[i,t]  = x[i,t]*w[i,t-1] + (1 - x[i,t])*w[i,t-1] 