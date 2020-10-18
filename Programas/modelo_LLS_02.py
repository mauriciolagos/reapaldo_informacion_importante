""" -*- coding: utf-8 -*-
   A microscopic model of the stock market: cycles, booms, and crashes
   Levy, Moshe and Levy, Haim and Solomon, Sorin
   Economics Letters
   1994 venv/
"""

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

tiempo = 250
agentes = 100
k = 10
Nt = 10000
d = 0.3
r = 0.01
sigma = 0.024
mu = 0.00001    #[0.00001, 0.01, 0.01, 0.2, 0.8] # factor de ruido o temperatura
mu1 = 0.1001
sigma1 = 0.024


def f_lls(x, wh, H, r, k):
    y = np.sum(np.log(wh*(1 + r)+x*wh*(H[:]-r)))/k
    return y


'''
def f_cero(v):
    for q in np.arange(len(v)):
        if v[q] < 0:
            v[q] = 0
    return v
'''

p = np.zeros(tiempo)
ph = np.zeros(agentes)
N = np.zeros((agentes, tiempo))
Nh = np.zeros(agentes)
w = np.zeros((agentes, tiempo))
wh = np.zeros(agentes)
x = np.zeros((agentes, tiempo))
x_op = np.zeros(agentes)
H = np.zeros(k)
datos = np.zeros((tiempo, 4)) # np.zeros((tiempo, len(mu) + 3))

w[:, 0] = 1000
p[0] = 4.60
N[:, 0] = Nt/agentes
for j in np.arange(k):
    H[j] = np.random.normal(mu1, sigma1)
bi = 0.01
bs = 0.99

for t in np.arange(tiempo):
    if t > 0:
        for i in np.arange(agentes):
            ph[i] = np.random.choice(H[:])
        ph[:] = ph[:]
        for i in np.arange(agentes):
            wh[i] = w[i, t-1] + N[i, t-1]*(ph[i]-p[t-1])
        for i in np.arange(agentes):
            x_op[i] = optimize.fminbound(f_lls, bi, bs, args=(wh[i], H, r, k))
        for i in np.arange(agentes):
            x[i, t] = x_op[i] + np.random.normal(mu, sigma)
        for i in np.arange(agentes):
            Nh[i] = x[i, t]*wh[i]/(ph[i])
        p[t] = np.sum(x[:, t]*wh[:])/Nt
        for i in np.arange(agentes):
            w[i, t] = w[i, t-1] + N[i, t-1]*(p[t] - p[t-1])
        for i in np.arange(agentes):
            N[i, t] = x[i, t]*w[i, t]/p[t]
        for j in np.arange(k):
            if j < k-1:
                H[j] = H[j+1]
            H[k-1] = (p[t] + d)sts/p[t-1]
            datos[t, 1] = H[k-1]
        Nt = np.sum(N[:, t])
        for i in np.arange(agentes):
            w[i, t] = w[i, t] + N[i, t]*d + (w[i, t] - N[i, t]*p[t])*r
        datos[t, 2] = Nt
        datos[t, 0] = t
        datos[t, 3] = p[t]
        print('_______ronda: ', t+1, '________________')



intro = ('  Programa que reproduce el Modelo Levy, Levy, Solomon publicado en \n'
         ' \'A microscopic model of the stock market: cycles, booms, and crashes\', \n'
         ' Economics Letters, 1994.\n')
intro += '__________________________________________________________________\n'
intro += '#Datos principales utilizados en el programa'
intro += ('\n   Agentes: {0}'.format(str(agentes)))
intro += ('\n   Iteracions: '+str(tiempo))
intro += ('\n   Largo memoria: '+str(k))
intro += '\n________________________________________________________________\n'
intro += 'Tiempo Precio Retorno[t] Acciones Totales'

np.savetxt('Datos/dt.txt', datos, fmt='%.2f', delimiter=' ') #, header=intro, comments=' ')


leg = ("Tiempo: "+str(tiempo))
leg += (", Agentes: "+str(agentes))
leg += (", Memoria:"+str(k))
titulo = ("Grafica tiempo v/s log(precio) con mu: "+str(mu))
titulo += (" y sigma: "+str(sigma))
plt.semilogy(np.arange(tiempo), datos[:, 3])
plt.xlabel("Tiempo (dias)")
plt.ylabel("Log(Precio)")
plt.title(titulo)
plt.legend(leg)
plt.show()

"""
direccion = ("/home/mauricio/Escritorio/Tesis/Implementacion/Graficas/Modelo_LLS_94/grafica")
direccion += "(tiempo="+str(tiempo) + ")(agente="+str(agentes) +  ")"
plt.savefig(direccion, format='png')
"""


print('                      _FIN_             ')
