#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:48:00 2022

@author: francesco

Script to compare the Constant Temperature formula to the Fermi Gas formula to the
best NLD fit. Used to choose which one fits best and should be used for the
extrapolation in counting.c for the Oslo method.
"""

import numpy as np
from systlib import chisquared
from iminuit import Minuit
import matplotlib.pyplot as plt
from iminuit.cost import LeastSquares

A = 166

def sigma(E, E1, a):
    U=E-E1
    sigma2 = 0.0146*A**(5/3)*(1+np.sqrt(1+4*a*U))/(2*a)
    return np.sqrt(sigma2)

def CT_model(E, params):
    E1 = params[0]
    T_CT = params[1]
    U=E-E1
    return np.exp(U/T_CT)/T_CT

def FG_model(E, params):
    E1 = params[0]
    a = params[1]
    U = E-E1
    return np.exp(2*np.sqrt(a*U))/(12*np.sqrt(2)*a**(1/4)*U**(5/4)*sigma(E, E1, a))

best_fits = np.load('data/generated/best_fits_FG.npy', allow_pickle = True)
best_nld = best_fits[0]
best_nld.clean_nans()
data_x = []
data_y = []
data_yerr = []
for idx, E in enumerate(best_nld.energies):
    if E > 1.5:
        data_x.append(E)
        data_y.append(best_nld.y[idx])
        data_yerr.append(best_nld.yerr[idx])

data_x = np.array(data_x)
data_y = np.array(data_y)
data_yerr = np.array(data_yerr)

#initial parameters
E1_I = -0.9
E0_I = -0.9
T_CT_I = 0.61
a_I = 1.8

fig, ax = plt.subplots()
ax.errorbar(data_x, data_y, data_yerr, label = 'oslo data')
models = ['CT', 'FG']
for model in models:
    
    if model == 'CT':
        model_func = CT_model
        label = 'CT model'
        params_init = [E0_I, T_CT_I]
        names = ['E1', 'T_CT']
        print('CT parameters:')
    else:
        model_func = FG_model
        label = 'FG model'
        params_init = [E1_I, a_I]
        names = ['E1', 'a']
        print('FG parameters:')
    
    least_squares = LeastSquares(data_x, data_y, data_yerr, model_func)
    m = Minuit(least_squares, params_init, name = names)
    m.migrad()
    m.hesse()   # accurately computes uncertainties
    
    print(m.params)
    chi2 = chisquared(model_func(data_x,m.values), data_y, data_yerr, reduced=False)
    print(f'{label} chi2: {chi2}')
    
    ax.plot(data_x, model_func(data_x, m.values), label = label)
    
ax.set_yscale('log')
ax.legend()
fig.show()

    