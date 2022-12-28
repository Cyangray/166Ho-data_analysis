#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:45:30 2022

@author: francesco
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['computer Modern'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})

#import list
a0 = -0.8433
a1 = 0.1221
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb
E_range = np.linspace(0, 20, 100)
BSR_constant = 2.5980e8 #u_N^2 MeV^2

NLD_pathstring = 'FG'
best_fits = np.load('data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
best_gsf = best_fits[1]
best_gsf.clean_nans()
best_gsf.delete_point(-1)
best_nld = best_fits[0]
best_nld.clean_nans()
curr_gsf = best_gsf

def expfunc(E,A,c):
    return A*np.exp(E*c)

p1_idx = 5
p2_idx = 21

#interpolate an exponential between two points of the gsf
x1 = best_gsf.x[p1_idx]
x2 = best_gsf.x[p2_idx]
y1 = best_gsf.y[p1_idx]
y1_err = best_gsf.yerr[p1_idx]
y2 = best_gsf.y[p2_idx]
y2_err = best_gsf.yerr[p2_idx]
c= np.log(y2/y1)/(x2-x1)
A = y2/np.exp(c*x2)

def exponential_area(x1,x2,y1,y2,E1,E2):
    c = np.log(y2/y1)/(x2-x1)
    A = y2*np.exp(-c*x2)
    return A*(np.exp(c*E2) - np.exp(c*E1))/c

def exponential_error(x1,x2,y1,y2,y1err,y2err,E1,E2):
    c = np.log(y2/y1)/(x2-x1)
    A = y2/np.exp(c*x2)
    dAdy1 = A/y1/(x2-x1) * ( (1 + c*E1 + c*x2)*np.exp(c*(E1)) - (1 + c*E2 - c*x2)*np.exp(c*(E2)) )
    dAdy2 = A*(  c*(np.exp(c*(E2)) - np.exp(c*(E1)))
             + 1/(x2-x1)*(np.exp(c*(E2)) - np.exp(c*(E1)))
             + c*((E2-x2)*np.exp(c*(E2)) - (E1-x2)*np.exp(c*(E1)))/(x2-x1)
        )/y2
    Area_err = dAdy1**2 * y1err**2 + dAdy2**2 * y2err**2
    return np.sqrt(Area_err)

def trapezoid_sigma2_two_points(x1,x2,y1err,y2err):
    return 0.25*(x2-x1)**2*(y2err**2 + y1err**2)

def trapezoid_error(x_array, y_array, yerr_array):
    sigma2 = 0
    for i in range(len(y_array)-1):
        sigma2 += trapezoid_sigma2_two_points(x_array[i], x_array[i+1], yerr_array[i], yerr_array[i+1])
    return np.sqrt(sigma2)
        
fig,ax = plt.subplots(figsize = (5.0, 3.75), dpi = 300)
ax.fill_between(best_gsf.x[p1_idx:p2_idx+1], expfunc(best_gsf.x[p1_idx:p2_idx+1], A, c), y2 = best_gsf.y[p1_idx:p2_idx+1], color = 'b', alpha = 0.4, label = 'Int. strength')
best_gsf.plot(ax=ax, color = 'k', marker = 'o', label = r'$^{166}$Ho GSF, this work')
ax.plot(E_range, expfunc(E_range,A,c), 'r-', label = 'E1 background')

E_cuts = best_gsf.x[p1_idx:p2_idx+1]
residuals = np.c_[ E_cuts, best_gsf.y[p1_idx:p2_idx+1] - expfunc(E_cuts, A, c)]
ax.errorbar(residuals[:,0], residuals[:,1]*2, yerr = best_gsf.yerr[p1_idx:p2_idx+1], marker = 'o',label = 'SR')
ax.set_yscale('log')
ax.set_xlabel(r'$E_\gamma$ (MeV)')
ax.set_ylabel('GSF (MeV$^{-3}$)')
ax.set_xlim([1,6])
ax.set_yticks([1e-8, 1e-7])
ax.set_ylim([1e-8,3e-7])
ax.legend(frameon = False)
fig.tight_layout()
fig.savefig('pictures/BSR_hildes_method_' + str(p1_idx) + '.pdf', format = 'pdf')
fig.show()


fits = []
fit2040 = [True, False]
for is_fit2040 in fit2040:
    if is_fit2040:
        p1_idx_int = 5
        p2_idx_int = 20
    else:
        p1_idx_int = 0
        p2_idx_int = 0
        
    if p1_idx_int and p2_idx_int and (p2_idx_int != p2_idx):  
        low_idx = p1_idx_int
        high_idx = p2_idx_int + 1
    else:
        low_idx = p1_idx
        high_idx = p2_idx + 1
    E1 = best_gsf.x[low_idx]
    E2 = best_gsf.x[high_idx-1]
    area_SR = BSR_constant*np.trapz( best_gsf.y[low_idx:high_idx], best_gsf.x[low_idx:high_idx])
    area_SR_err = BSR_constant*trapezoid_error(best_gsf.x[low_idx:high_idx], best_gsf.y[low_idx:high_idx], best_gsf.yerr[low_idx:high_idx])
    area_background = BSR_constant*exponential_area(x1,x2,y1,y2,E1,E2)
    area_background_err = BSR_constant*exponential_error(x1,x2,y1,y2,y1_err,y2_err,E1,E2)
    area = (area_SR - area_background)
    area_err = np.sqrt(area_SR_err**2 + area_background_err**2)
    fit = [area, area_err]
    fits.append(fit)
    print(area)
    print(area_err)

output = np.array([[fits[0][0], fits[1][0]],
                   [fits[0][1], fits[1][1]]])
header = 'BSR_2040, BSR_0010 for p1_idx = ' + str(p1_idx) + '. Second line:uncertainties'
np.savetxt('data/generated/Hildes_metode_' + str(p1_idx) + '.txt', output, header = header)
    




