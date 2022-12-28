#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:26:05 2021

@author: francesco, updated 21st March 2022

Script to loop through the folders made with run_cnt_nrm.py, and writes for
each gsf and nld a e1strength, m1strength and .tab file to be used as input by talys,
possibly in Saga. In order to keep the number of nld-gsf pairs low, only the pairs
with a chi2 score between chimin and chimin+2 will be translated into talys-input files.
(To calculate all, just put chimin2 to a very big number)
In order to extrapolate for the GDR, a high-energy extrapolation of the GSF is attached
"""

import os
import numpy as np
from systlib import make_E1_M1_files, make_TALYS_tab_file, load_known_gsf, make_E1_M1_files_simple, D2rho, drho, GLO_hybrid_arglist, SLO_arglist

#paths
NLD_pathstring = 'FG'
master_folder = '166Ho-saga-original/'
root_folder = '/home/francesco/Documents/163Dy-alpha-2018/Fittings/'
data_folder = root_folder + 'data/'
dataset_folder = root_folder + 'Make_dataset/'

#constants. Don't play with these
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb

#Don't modify unless you know what you're doing
A = 166
Z = 67
L1min = 7
L1max = 18
nrhos = 51
blist = np.linspace(0.75,1.25,nrhos)
target_spin = 3.5
Sn = 6.243
a0 = -0.8433
a1 = 0.1221
spin_cutoff_low = 5.546
spin_cutoff_high = 6.926
cutoff_unc = 0
D0 = 4.35
D0_err = 0.15
Gg_mean = 84
Gg_sigma = 5
Gglist = np.linspace(73,97,25)

rho_Sn_err_up = drho(target_spin, spin_cutoff_high, spin_cutoff_high*cutoff_unc, D0, D0_err)
rho_Sn_err_down = drho(target_spin, spin_cutoff_low, spin_cutoff_low*cutoff_unc, D0, D0_err)
rho_lowerlim = D2rho(D0, target_spin, spin_cutoff_low)
rho_upperlim = D2rho(D0, target_spin, spin_cutoff_high)
rho_mean = (rho_lowerlim - rho_Sn_err_down + rho_upperlim + rho_Sn_err_up)/2
rho_sigma = rho_upperlim + rho_Sn_err_up - rho_mean
base_spin_cutoff = (spin_cutoff_low + spin_cutoff_high)/2
base_rho = D2rho(D0, target_spin,base_spin_cutoff)
rho_flat_distr = True

#load the total fit from method 2(A) and then select only the values above a certain energy (e.g. 6.5 MeV)
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

nlds = np.load('data/generated/nlds_' + NLD_pathstring + '.npy', allow_pickle = True)
gsfs = np.load('data/generated/gsfs_' + NLD_pathstring + '.npy', allow_pickle = True)

#load best fits
best_fits = np.load('data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
best_gsf = best_fits[1]
best_gsf.clean_nans()
best_gsf.delete_point(-1)
best_nld = best_fits[0]

#find chimin (common for nld, gsfs)
chimin = best_gsf.chi2

#calculate chimin+2
chimin2 = chimin + 2

#make new gsf list with only chi<chimin+2 (count elements!)
gsfs_filtered = []
for gsf in gsfs:
    if gsf.chi2<=chimin2:
        gsfs_filtered.append(gsf)

#begin the main loop
try:
    os.mkdir(dataset_folder + '/' + master_folder)
except:
    pass
for count, gsf in enumerate(gsfs_filtered):
    L1 = gsf.L1
    L2 = gsf.L2
    rho = gsf.rho
    Gg = gsf.Gg
    
    #retrieve the nld from 166Ho-database using L1 L2 and rho, build Ho.tab, put in folder
    bstr_int = "{:.2f}".format(gsf.b)
    bstr = bstr_int.translate(bstr_int.maketrans('','', '.'))
    new_rho_str = '{:.6f}'.format(rho)
    new_dir_rho = bstr + '-' + str(int(rho)) + '/'
    new_dir_L1_L2 = 'L1-'+L1+'_L2-'+L2 + '/'
    new_dir_Gg = str(int(Gg)) + '/'
    nld_dir = new_dir_rho + new_dir_L1_L2
    Gg_dir = nld_dir + new_dir_Gg
    os.makedirs(dataset_folder + master_folder + nld_dir, exist_ok = True)
    os.system('cp ' + 'Backup/Ho.tab '+ dataset_folder + master_folder + nld_dir + 'Ho.tab')
    make_TALYS_tab_file(dataset_folder + master_folder + nld_dir + 'Ho.tab', dataset_folder + '166Ho-database_' + NLD_pathstring + '/' + nld_dir + 'talys_nld_cnt.txt', A, Z)
    
    #create Gg folder, put E1, M1 and talys input file there
    os.makedirs(dataset_folder + master_folder + Gg_dir)
    E_tal, E1_tal, M1_tal = make_E1_M1_files(dataset_folder + '166Ho-database_' + NLD_pathstring + '/' + Gg_dir,
                                      A, 
                                      Z, 
                                      a0, a1, 
                                      M1 = M1pars, 
                                      target_folder = dataset_folder + master_folder + Gg_dir, 
                                      high_energy_interp = Ho165mat)
    
    #copy input file to working folder
    os.system('cp ' + dataset_folder + '166Ho ' + dataset_folder + master_folder + Gg_dir + 'input.txt')
    os.chdir(dataset_folder + master_folder + Gg_dir)
    os.system('tar -czf ' + dataset_folder + 'Gg_input.tar.gz ./')
    os.system('rm ./*')
    os.system('mv ' +  dataset_folder + 'Gg_input.tar.gz ./')
    os.chdir(root_folder)
    
    #Don't tar first, count how many gsf pairs I have
    if count%100 == 0:
        print('.')

#zip rho directories
counter = 0
highest_number_of_Gg_folders = 0
for rho_dir in os.listdir(dataset_folder + master_folder):
    for L1L2 in os.listdir(dataset_folder + master_folder + rho_dir):
        counter += 1
        zipping_dir = dataset_folder + master_folder + rho_dir + '/' + L1L2
        os.chdir(zipping_dir)
        var = os.popen('ls | wc -l').read()
        var2 = int(var)
        if var2 > highest_number_of_Gg_folders:
            highest_number_of_Gg_folders = var2
        os.system('tar -czf ' + dataset_folder + 'L1_L2.tar.gz ./')
        os.system('rm -r ./*')
        os.system('mv ' + dataset_folder + 'L1_L2.tar.gz ./')

print(f'Highest number of Gg folders in a L1L2-folder: {highest_number_of_Gg_folders}')
print(f'antall L1L2-mapper: {counter}')
os.chdir(root_folder)
