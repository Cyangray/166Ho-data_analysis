#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 17:15:38 2021

@author: francesco, updated December 28th 2022

Draw nld from the nld_whole.txt file produced from make_nlds_gsfs_lists.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from readlib import readldmodel
from systlib import import_ocl, import_ocl_fermi, D2rho, chisquared

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['computer Modern'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})

#parameters. play with these
L1min = 7
L2max = 18
enbin = 22

hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb

#Don't modify unless you know what you're doing
nrhos = 51
blist = np.linspace(0.75,1.25,nrhos)
Gglist = np.linspace(75,95,25)
target_spin = 3.5
D0 = 4.35
D0_err = 0.15
spin_cutoff_low = 5.546
spin_cutoff_high = 6.926
rho_Sn_err_up = 120000
rho_Sn_err_down = 80000
rho_lowerlim = D2rho(D0, target_spin, spin_cutoff_low)
rho_upperlim = D2rho(D0, target_spin, spin_cutoff_high)
rho_mean = (rho_lowerlim - rho_Sn_err_down + rho_upperlim + rho_Sn_err_up)/2
rho_sigma = rho_upperlim + rho_Sn_err_up - rho_mean
chi2_lim = [9,13]

Sn = 6.243
a0 = -0.8433
a1 = 0.1221
Gg_mean = 84
Gg_sigma = 0.5

sig = 0.341
limit_points = np.linspace(0.0,1.0,21)

NLD_pathstrings = ['FG']
fig_labs = ['a)', 'b)']

for NLD_pathstring in NLD_pathstrings:
    
    
    ranges = [[1,7], [np.log10(1e-8),np.log10(4e-7)]]
    database_path = 'Make_dataset/166Ho-database_' + NLD_pathstring + '/'
    
    #load best fits
    best_fits = np.load('data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
    best_nld = best_fits[0]
    best_nld.clean_nans()
    
    extr_path = best_nld.path[:-10] + '/fermigas.cnt'
    extr_mat = import_ocl_fermi(extr_path,a0,a1)
    
    extr_vals = []
    nld_vals = []
    nld_errvals = []
    for idx, E in enumerate(best_nld.energies):
        if E > 1.5:
            idx2 = np.argwhere(extr_mat[:,0] == E)[0,0]
            extr_vals.append(extr_mat[idx2,1])
            nld_vals.append(best_nld.y[idx])
            nld_errvals.append(best_nld.yerr[idx])
            
    chisq = chisquared(extr_vals, nld_vals, nld_errvals, DoF=1, method = 'linear',reduced=True)
    
    #import experimental nld and gsf
    nld_mat = np.genfromtxt('data/generated/nld_' + NLD_pathstring + '_whole.txt', unpack = True).T
    #delete rows with nans
    nld_mat = nld_mat[~np.isnan(nld_mat).any(axis=1)]
    
    #import known levels
    known_levs = import_ocl('data/rholev.cnt',a0,a1)
    
    
    #import TALYS calculated NLDs
    TALYS_ldmodels = [readldmodel(67, 166, ld, 1, 1, 1) for ld in range(1,7)]
    
    #start plotting
    cmap = matplotlib.cm.get_cmap('YlGnBu')
    fig0,axs = plt.subplots(nrows = 2, ncols = 1, figsize = (5.0, 3.75*2), dpi = 300)
    [ax0,ax1] = axs.flatten()
    
    chi2_lim_energies = [known_levs[int(chi2_lim[0]),0], known_levs[int(chi2_lim[1]),0]]
    ax0.axvspan(chi2_lim_energies[0], chi2_lim_energies[1], alpha=0.2, color='red',label='Fitting interval')
    ax0.plot(known_levs[:,0],known_levs[:,1],'k-',label='Known levels')
    
    #Plot experiment data
    val_matrix = nld_mat
    staterrs = best_nld.yerr
    
    #ax0
    ax0.errorbar(Sn, rho_mean,yerr=rho_sigma,ecolor='g',linestyle=None, elinewidth = 4, capsize = 5, label=r'$\rho$ at Sn')
    ax0.fill_between(val_matrix[:,0], val_matrix[:,2], val_matrix[:,-2], color = 'c', alpha = 0.2, label=r'2$\sigma$ conf.')
    ax0.fill_between(val_matrix[:,0], val_matrix[:,3], val_matrix[:,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
    ax0.errorbar(val_matrix[:,0], val_matrix[:,1],yerr=val_matrix[:,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
    
    #ax1
    ax1.errorbar(Sn, rho_mean,yerr=rho_sigma,ecolor='g',linestyle=None, elinewidth = 4, capsize = 5)
    ax1.fill_between(val_matrix[:,0], val_matrix[:,2], val_matrix[:,-2], color = 'c', alpha = 0.2)
    ax1.fill_between(val_matrix[:,0], val_matrix[:,3], val_matrix[:,-3], color = 'b', alpha = 0.2)
    ax1.errorbar(val_matrix[:,0], val_matrix[:,1],yerr=val_matrix[:,-1], fmt = '.', color = 'b', ecolor='b')
    
    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$E_x$ [MeV]')
    
    #plot TALYS nld
    stls = ['-','--','-.',':','-','--','-.',':']
    for i, TALYS_ldmodel in enumerate(TALYS_ldmodels):
        if i<3:
            col = 3
        else:
            col = 5
        ax1.plot(TALYS_ldmodel[:,0],TALYS_ldmodel[:,1], color = cmap(col/6), linestyle = stls[i], alpha = 0.8, label = 'ldmodel %d'%(i+1))
    
    ax0.plot(extr_mat[9:,0], extr_mat[9:,1], color = 'k', linestyle = '--', alpha = 1, label = 'FG extrap.')
    
    #plot
    energies = np.linspace(0, 20, 1000)
    for ax, lab in zip([ax0,ax1], fig_labs):
        ax.set_ylabel(r'NLD [MeV$^{-1}$]')
        ax.set_xlim(-0.8,7.5)
        ax.set_ylim(10**ranges[0][0],10**ranges[0][1])
        ax.legend(loc = 'lower right', ncol = 1, frameon = False)
        ax.text(0.05, 0.95, lab, fontsize='large', verticalalignment='center', fontfamily='serif', transform = ax.transAxes)
    fig0.tight_layout()
    fig0.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig0.axes[:-1]], visible=False)
    fig0.tight_layout()
    fig0.show()
    
    fig0.savefig('pictures/nld_'+NLD_pathstring+'_double.pdf', format = 'pdf')
    