#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:58:32 2022

@author: francesco, updated December 28th 2022

Code plotting the MACS of 165Ho from MACS_whole.txt generated in make_nlds_gsfs_lists.py
and compares it to TALYS predictions, and other libraries
"""
import numpy as np
import matplotlib.pyplot as plt
from readlib import readreaclib, nonsmoker, rate2MACS, readastro, MACS2rate
import pandas as pd
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['computer Modern'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})

suffix = 'FG'

#inputs
Z= 67
nucleus = Z                     #isotope from which to extract strength function. one, or a series of names, or Zs
A = 165                         #mass number of isotope
omp = [1,2]                     #three options: 0, 1 and 2. 0 for no arguments, 1 for jlmomp-y and 2 for localomp-n
strength = [1,2,3,4,5,6,7,8]    #Which strength function to read - numbers from 1 to 8
nld = [1,2,3,4,5,6]
mass = [1,2,3]
rates = False
Ho165_mass = 164.930329116      #in a.u.

#constants
k_B = 8.617333262145e1          #Boltzmann konstant, keV*GK^-1

#useful arrays
T9range = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
T9extrange = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
keVrange = T9extrange*k_B
farger=np.array([[199,176,247],
        [117,98,198],
        [23,12,131],
        [182,195,244]
        ])/255.

#Make TALYS uncertainty span
dummyastro = readastro(nucleus, A, nld[0], mass[0], strength[0], omp[0])
if rates:    
    val_index = 1
else:
    val_index = 2
max_val = np.zeros_like(dummyastro[:,val_index])
min_val = dummyastro[:,val_index].copy()
for ld in nld:
    for m in mass:
        for s in strength:
            for o in omp:
                currentastro = readastro(nucleus, A, ld, m, s, o)
                alpha = 0.2
                for i,val in enumerate(currentastro[:,val_index]):
                    if val > max_val[i]:
                        max_val[i] = val
                    if val < min_val[i]:
                        min_val[i] = val
                
#Import other models
val_reaclib = nonsmoker(readreaclib(nucleus, A, reaclib_file = 'data/ncapture/reaclib'), T9range) #rates
other_libs_df = pd.read_csv('data/ncapture/MACS_from_libs.txt', sep='	', header = 1)
models = [other_libs_df[other_libs_df['modn']==j] for j in range(1,6)]
kadonis = np.loadtxt('data/ncapture/kadonis.txt')
mat_bruslib = np.loadtxt('data/ncapture/bruslib-165Ho')
                
if rates:
    #Import Oslo data, statistical and systematic errors
    Oslo_mat = np.genfromtxt('data/generated/ncrates_' + suffix + '_whole.txt', unpack = True).T
    x_Oslo = Oslo_mat[:,0]
    Oslo_stat = np.loadtxt('data/generated/rate_' + suffix + '_stats.txt')
    
    #import other models
    x_reaclib = T9range
    x_bruslib = mat_bruslib[:,0]
    val_bruslib = mat_bruslib[:,1]
    x_kadonis = kadonis[:,0]/k_B
    y_kadonis = MACS2rate(kadonis[:,1], Ho165_mass, x_kadonis)
else:
    #Import Oslo data, statistical and systematic errors
    Oslo_mat = np.genfromtxt('data/generated/MACS_' + suffix + '_whole.txt', unpack = True).T
    x_Oslo = Oslo_mat[:,0]*k_B #keV
    Oslo_stat = np.loadtxt('data/generated/MACS_' + suffix + '_stats.txt')
    
    #import other models
    x_reaclib = T9range*k_B
    val_reaclib = rate2MACS(val_reaclib,Ho165_mass,T9range)
    x_bruslib = mat_bruslib[:,0]*k_B
    val_bruslib = rate2MACS(mat_bruslib[:,1],Ho165_mass,mat_bruslib[:,0])
    x_kadonis = kadonis[:,0]
    y_kadonis = kadonis[:,1]

#plot figure
my_cmap = matplotlib.cm.get_cmap('YlGnBu')
#my_cmap = matplotlib.cm.get_cmap('Blues')
#my_cmap = matplotlib.colors.ListedColormap(farger)

fig, ax = plt.subplots(figsize = (5.0, 3.75), dpi = 400)
ax.fill_between(x_Oslo, min_val, max_val, color = my_cmap(1/4), alpha = 1, label = 'TALYS unc. span')
lowerline_2s = Oslo_mat[:,1] - np.sqrt((Oslo_mat[:,1] - Oslo_mat[:,2])**2 + (Oslo_stat[:,1] - Oslo_stat[:,2])**2)
upperline_2s = Oslo_mat[:,1] + np.sqrt((Oslo_mat[:,5] - Oslo_mat[:,1])**2 + (Oslo_stat[:,3] - Oslo_stat[:,1])**2)
lowerline_1s = Oslo_mat[:,1] - np.sqrt((Oslo_mat[:,1] - Oslo_mat[:,3])**2 + (Oslo_stat[:,1] - Oslo_stat[:,2])**2)
upperline_1s = Oslo_mat[:,1] + np.sqrt((Oslo_mat[:,4] - Oslo_mat[:,1])**2 + (Oslo_stat[:,3] - Oslo_stat[:,1])**2)
ax.fill_between(x_Oslo, lowerline_2s, upperline_2s, color = my_cmap(2/4), alpha= 1, label = r'Oslo data, $2\sigma$')
ax.fill_between(x_Oslo, lowerline_1s, upperline_1s, color = my_cmap(3/4), alpha= 1, label = r'Oslo data, $1\sigma$')
ax.plot(x_Oslo, Oslo_mat[:,1], color = my_cmap(4/4), linestyle ='-', label = 'Oslo data')
ax.plot(x_kadonis, y_kadonis, color = my_cmap(4/4), linestyle ='--', label = 'Kadonis')

for i, model in enumerate(models):
    label = model['model'].iloc[0]
    label = label[:-1]
    matr = model[['kT','MACS']].to_numpy()
    if rates:
        x_libs = matr[:,0]/k_B
        y_libs = MACS2rate(matr[:,1], Ho165_mass, x_libs)
    else:
        x_libs = matr[:,0]
        y_libs = matr[:,1]
    if i==0:
        ax.plot(x_libs, y_libs, color = my_cmap(5/8), linestyle = '-.', label=label)

ax.plot(x_reaclib, val_reaclib, color = 'k', linestyle = '--',label = 'JINA REACLIB')
ax.plot(x_bruslib, val_bruslib, color = 'k', linestyle = ':', label = 'BRUSLIB')

if rates:
    #ax.set_title(r'rates for '+ ToLatex(str(A) + Z2Name(nucleus)) + '$(n,\gamma)$')
    ax.set_xlabel(r'$T$ [GK]')
    ax.set_ylabel(r'$(n,\gamma)$ rate [cm$^{3}$mol$^{-1}$s$^{-1}$]')
    #ax.grid()
    ax.set_yscale('log')
    ax.set_xlim([-0.1,110/k_B])
    ax.set_ylim([3e6,1.3e9])
    ax.legend(ncol=2)
else:
    #ax.set_title('MACS for '+ ToLatex(str(A) + Z2Name(nucleus)) + '$(n,\gamma)$')
    ax.set_xlabel(r'$k_B$ $T$ [keV]')
    ax.set_ylabel('MACS [mb]')
    #ax.grid()
    ax.set_yscale('log')
    ax.set_xlim([-2,110])
    ax.set_ylim([250,40e3])
    ax.legend(ncol=2)
    
fig.tight_layout()
fig.savefig('pictures/MACS_165Ho_' + suffix)#, transparent = True, format = 'svg')
fig.show()