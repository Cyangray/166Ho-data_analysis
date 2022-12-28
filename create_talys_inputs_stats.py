#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:26:05 2021

@author: francesco, updated December 28th 2022
Script that runs TALYS for the best fitting nld and gsf, and the four combinations
of low and high error limit. Then writes the highest and lowest calculated
neutron-capture rate and MACS to file.
"""

import os
import numpy as np
from systlib import make_E1_M1_files, make_TALYS_tab_file, load_known_gsf, make_E1_M1_files_simple, D2rho, GLO_hybrid_arglist, SLO_arglist, make_TALYS_tab_file_from_vals,make_TALYS_tab_files_linear_calc
from readlib import readastro_path
import matplotlib.pyplot as plt
from utils import Z2Name

#paths
talys_path = '/home/francesco/talys/'
master_folder = '166Ho-stats/'
root_folder = '/home/francesco/Documents/163Dy-alpha-2018/Fittings/'
data_folder = root_folder + 'data/'
dataset_folder = root_folder + 'Make_dataset/'
NLD_pathstring = 'FG'
stats_folder = root_folder + 'TALYS_stats_' + NLD_pathstring + '/'

Sn = 6.243
A = 166
Z = 67
k_B = 8.617333262145e1 #Boltzmann konstant, keV*GK^-1
NLDa = 18.277
Eshift = -0.949
Estop = Sn-0.5

nlderrup = [0,1]
gsferrup = [0,1]

#OR load the total fit from method 2(A) and then select only the values above a certain energy (e.g. 6.5 MeV)
high_energies = np.linspace(0.1, 20, 1000)
x_values_cont = np.linspace(0.1, 20, 1000)

GDR1 = [0.59, 12.40,3.50,323]
GDR2 = [0.59, 14.80,1.82,183]
PDR = [6.07,1.89,5.0]
M1pars = SR = [3.20,1.00,0.40]
PDR7 = [6.07,1.89,5.0/7]
high_Ho165_vals = GLO_hybrid_arglist(x_values_cont, GDR1) + GLO_hybrid_arglist(x_values_cont, GDR2) + SLO_arglist(x_values_cont, PDR) + SLO_arglist(x_values_cont, SR)
M1pars.extend(PDR7)

indexes_to_delete = np.argwhere(high_energies<6.5) #delete values less than 6.5 MeV
Ho165energies = np.delete(high_energies, indexes_to_delete, 0)
Ho165y = np.delete(high_Ho165_vals, indexes_to_delete, 0)
Ho165mat = np.c_[Ho165energies, Ho165y]

#load best fits
best_fits = np.load(data_folder + 'generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
best_gsf = best_fits[1]
best_gsf.clean_nans()
best_gsf.delete_point(-1)
best_nld = best_fits[0]
best_nld.clean_nans()
spinpars = {"sigma2_disc": [0.2,4.5], "sigma2_high": [Sn,best_nld.spin_cutoff]}
best_rate = best_fits[2]
TkB = best_rate.T * k_B
gsf_energies = best_gsf.energies
nld_energies = best_nld.energies


os.makedirs(stats_folder, exist_ok = True)

count = 0

rates = [[[], []],
         [[], []]
         ]
MACSs = [[[], []],
         [[], []]
         ]

#make base tab file
nld_vals = best_nld.y
nld_folder = stats_folder + 'nld_base/'
os.makedirs(nld_folder, exist_ok = True)
os.system('cp ' + root_folder + 'Backup/Ho.tab '+ nld_folder + 'Ho.tab')
make_TALYS_tab_file(nld_folder + 'Ho.tab', best_nld.path[:-10] + 'talys_nld_cnt.txt', A, Z)

#copy Ho.tab into TALYS
os.system('cp ' + nld_folder + 'Ho.tab ' + talys_path + 'structure/density/ground/goriely/Ho.tab')

#make base gsf files
gsf_vals = best_gsf.y
gsf_folder = nld_folder + 'gsf_base/'
os.makedirs(gsf_folder, exist_ok = True)
E_tal, E1_tal, M1_tal = make_E1_M1_files_simple(gsf_energies,
                                         gsf_vals,
                                         A, 
                                         Z, 
                                         M1 = M1pars, 
                                         target_folder = gsf_folder, 
                                         high_energy_interp = Ho165mat)

#copy talys input to folder
os.system('cp ' + dataset_folder + '166Ho ' + gsf_folder + 'input.txt')

#run talys
os.chdir(gsf_folder)
os.system(talys_path + 'talys <input.txt> output.txt')

#read and save astrorates
base_astrorate_output = readastro_path('astrorate.g')
base_rate = np.c_[base_astrorate_output[:,0],base_astrorate_output[:,1]]
base_MACS = np.c_[base_astrorate_output[:,0]*k_B,base_astrorate_output[:,2]]

#prepare for stat errors calc
#make tab files for up and down errors in nld
os.chdir(stats_folder)
os.system('cp ' + root_folder + 'Backup/Ho.tab '+ stats_folder + Z2Name(Z) + '_up.tab')
os.system('cp ' + root_folder + 'Backup/Ho.tab '+ stats_folder + Z2Name(Z) + '_down.tab')
make_TALYS_tab_files_linear_calc(best_nld, Sn, A, Z)

os.chdir(root_folder)
up_or_down = ['up', 'down']
for i in nlderrup:
    
    #copy either the up or the down file into talys
    os.system('cp ' + stats_folder + Z2Name(Z) + '_' + up_or_down[i] + '.tab ' + talys_path + 'structure/density/ground/goriely/Ho.tab')
    
    
    nld_folder = stats_folder + 'nld' + str(i) + '/'
    os.makedirs(nld_folder, exist_ok = True)
    gsf_folders = ['','']
    for j in gsferrup:
        
        if j:
            gsf_vals = best_gsf.y+best_gsf.yerr
        else:
            gsf_vals = best_gsf.y-best_gsf.yerr
        
        gsf_folders[j] = nld_folder + 'gsf' + str(j) + '/'
        os.makedirs(gsf_folders[j], exist_ok = True)
        E_tal, E1_tal, M1_tal = make_E1_M1_files_simple(gsf_energies,
                                                 gsf_vals,
                                                 A, 
                                                 Z, 
                                                 M1 = M1pars, 
                                                 target_folder = gsf_folders[j], 
                                                 high_energy_interp = Ho165mat)
        os.system('cp ' + dataset_folder + '166Ho ' + gsf_folders[j] + 'input.txt')

    for j in gsferrup:
        
        os.chdir(gsf_folders[j])
        os.system(talys_path + 'talys <input.txt> output.txt')
        
        curr_astrorate_output = readastro_path('astrorate.g')
        curr_rate = np.c_[curr_astrorate_output[:,0],curr_astrorate_output[:,1]]
        curr_MACS = np.c_[curr_astrorate_output[:,0]*k_B,curr_astrorate_output[:,2]]
        os.chdir(root_folder)
        rates[i][j]=curr_rate
        MACSs[i][j]=curr_MACS
        print(f'done with nlderrup: {i} and gsferrup {j}')

#plot the four rates together with labels
fig, axs = plt.subplots(nrows = 1, ncols = 2)
for i in nlderrup:
    for j in gsferrup:
        axs[0].plot(rates[i][j][:,0], rates[i][j][:,1], label = str(i) + str(j))
        axs[1].plot(MACSs[i][j][:,0], MACSs[i][j][:,1], label = str(i) + str(j))

axs[0].plot(best_rate.T, best_rate.ncrate, label = 'best rate')
axs[1].plot(best_rate.T*k_B, best_rate.MACS, label = 'best MACS')
axs[0].plot(base_rate[:,0], base_rate[:,1], 'k-', label = 'local TALYS')
axs[1].plot(base_MACS[:,0], base_MACS[:,1], 'k-', label = 'local TALYS')


for ax in axs:
    ax.set_yscale('log')
plt.show()
#Replace original Ho.tab in TALYS
os.system('cp ' + root_folder + 'Backup/Ho.tab ' + talys_path + 'structure/density/ground/goriely/Ho.tab')


least_rates = np.zeros_like(best_rate.ncrate)
most_rates = np.zeros_like(best_rate.ncrate)
least_MACSs = np.zeros_like(best_rate.MACS)
most_MACSs = np.zeros_like(best_rate.MACS)
for index in range(len(TkB)):
    
    least_rate = min([rates[i][j][index,1] for i in nlderrup for j in gsferrup])
    most_rate = max([rates[i][j][index,1] for i in nlderrup for j in gsferrup])
    least_MACS = min([MACSs[i][j][index,1] for i in nlderrup for j in gsferrup])
    most_MACS = max([MACSs[i][j][index,1] for i in nlderrup for j in gsferrup])
    least_rates[index] = least_rate
    most_rates[index] = most_rate
    least_MACSs[index] = least_MACS
    most_MACSs[index] = most_MACS

MACS_header = '# TkB (keV), bestfit, lower, upper'
rate_header = '# T (GK), bestfit, lower, upper'
MACS_statsmat = np.c_[TkB,best_rate.MACS,least_MACSs, most_MACSs]
rate_statsmat = np.c_[best_rate.T, best_rate.ncrate,least_rates,most_rates]
np.savetxt('data/generated/MACS_FG_stats.txt',MACS_statsmat,header = MACS_header)
np.savetxt('data/generated/rate_FG_stats.txt',rate_statsmat,header = rate_header)





