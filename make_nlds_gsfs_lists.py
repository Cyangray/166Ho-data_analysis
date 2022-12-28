#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Oct 14 09:49:41 2021

@author: francesco, updated December 28th 2022

To run this code, you have to have run run_cnt_nrm.py in Make_dataset/

Script looping thorugh all the produced rhopaw.cnt, strength.nrm (and astrorate.g
if astro = True), and:
1) create numpy arrays of nld and gsf (and astrorate) objects, storing all information
    produced by counting.c, normalization.c (and TALYS). These will be saved as
    nlds.npy, gsfs.npy and astrorates.npy
2) To each nld, gsf or astrorate is associated a chi2 score, telling how well
    the nld fits to the known levels in the two fitting intervals chi_lim and chi_lim2.
3) For each energy bin of nld, gsf and MACS, the uncertainty is found checking
    graphically where the chi2 parabola crosses the chi2+1 line. The results are
    saved to text files.
4) Save the nld-gsf couple with the least chi2 to best_fits.npy
5) If plot_chis = True, it plots the parabolas for one specific energy bin of 
    the nld and the gsf, + astrorate/MACS if astro = True
'''

import numpy as np
import matplotlib.pyplot as plt
from systlib import import_ocl, nld, gsf, import_Bnorm, astrorate, import_Anorm_alpha, chisquared, flat_distr_chi2_fade, calc_errors_chis, calc_errors_chis_MACS, D2rho, rho2D, drho

#paths
NLD_pathstring = 'FG'  
database_path = '/home/francesco/Documents/163Dy-alpha-2018/Fittings/Make_dataset/166Ho-database_' + NLD_pathstring + '/'

#constants. Don't play with these
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb

#Don't modify unless you know what you're doing
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
cutoff_unc = 0.00

#data from Mughabghab
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

#Play with these parameters
#region of the nld where to evaluate the chi squared test
chi2_lim = [9,13]  #Fitting interval. Limits are included.
method = 'linear'

#some switches
load_lists = False
plot_chis = True
astro = False
energy_bin = 30
temp_bin = 7

'''
Initializing main loop. 
loop idea: loop through all L1 and L2 (the lower region to fit to in counting.c),
rho and Gg. Each such parameter combination points to a specific rhopaw.cnt 
produced in run_counting.py. Imports the NLD and the GSF, calculate the chi 
squared test to the region in chi2_lim, save all the parameters in the nld and 
gsf objects, and the objects in lists.
'''

Ho166_nld_lvl = import_ocl('data/rholev.cnt',a0,a1)
chi2_lim_e = [Ho166_nld_lvl[int(chi2_lim[0]),0], Ho166_nld_lvl[int(chi2_lim[1]),0]]

if load_lists:
    nlds = np.load('data/generated/nlds_' + NLD_pathstring + '.npy', allow_pickle = True)
    gsfs = np.load('data/generated/gsfs_' + NLD_pathstring + '.npy', allow_pickle = True)
    if astro:
        ncrates = np.load('data/generated/ncrates_' + NLD_pathstring + '.npy', allow_pickle = True)
else:
    #beginning the big nested loop
    gsfs = []
    nlds = []
    if astro:
        ncrates = []
    for b in blist:
        bstr_int = "{:.2f}".format(b)
        bstr = bstr_int.translate(bstr_int.maketrans('','', '.'))
        new_spincutoff = b*base_spin_cutoff
        new_rho = D2rho(D0, target_spin, new_spincutoff)
        new_drho = drho(target_spin, new_spincutoff, new_spincutoff*cutoff_unc, D0, D0_err, rho = new_rho)
        new_D = D0
        new_rho_str = '{:.6f}'.format(new_rho)
        new_D_str = '{:.6f}'.format(new_D)
        new_dir_rho = bstr + '-' + str(int(new_rho))
        print('Bstr: %s'%bstr_int) #print something to show the progression
        
        for L1n in range(L1min,L1max):
            L1 = str(L1n)
            if L1n == 1:
                L2_skip = 2
            else:
                L2_skip = 1
            for L2n in range(L1n + L2_skip, L1max):
                L1 = str(L1n)
                L2 = str(L2n)
                new_dir_L1_L2 = 'L1-'+L1+'_L2-'+L2
                curr_nld = nld(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/rhopaw.cnt',a0 = a0, a1 = a1, is_ocl = True)
                Anorm, alpha = import_Anorm_alpha(database_path +  new_dir_rho + '/' + new_dir_L1_L2 + '/alpha.txt')
                
                #calculate the reduced chi2
                lvl_values = Ho166_nld_lvl[chi2_lim[0]:(chi2_lim[1]+1),1]
                ocl_values = curr_nld.y[chi2_lim[0]:(chi2_lim[1]+1)]
                ocl_errs = curr_nld.yerr[chi2_lim[0]:(chi2_lim[1]+1)]
                chi2 = chisquared(lvl_values, ocl_values, ocl_errs, DoF = 1, method = method, reduced=False)
                
                #store values in objects
                curr_nld.L1 = L1
                curr_nld.L2 = L2
                curr_nld.Anorm = Anorm
                curr_nld.alpha = alpha
                curr_nld.rho = new_rho
                curr_nld.drho = new_drho
                curr_nld.b = b
                curr_nld.spin_cutoff = new_spincutoff
                curr_nld.D0 = new_D
                if rho_flat_distr:
                    curr_nld.chi2 = chi2 + flat_distr_chi2_fade(rho_upperlim, rho_lowerlim, [rho_Sn_err_down,rho_Sn_err_up], new_rho)
                else:
                    curr_nld.chi2 = chi2 + ((rho_mean - new_rho)/rho_sigma)**2
                nlds.append(curr_nld)
                
                for Gg in Gglist:
                    Ggstr = str(int(Gg))
                    curr_gsf = gsf(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/' + Ggstr + '/strength.nrm', a0 = a0, a1 = a1, is_sigma = False, is_ocl = True)
                    Bnorm = import_Bnorm(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/' + Ggstr + '/input.nrm')
                    if astro:
                        found_astro = False
                        try:
                            curr_ncrate = astrorate(database_path + new_dir_rho + '/' + new_dir_L1_L2 + '/' + Ggstr + '/astrorate.g')
                            objs = [curr_gsf, curr_ncrate]
                            found_astro = True
                        except:
                            objs = [curr_gsf]
                    else:
                        objs = [curr_gsf]
    
                    #store values in objects
                    for el in objs:
                        el.L1 = L1
                        el.L2 = L2
                        el.Anorm = Anorm
                        el.Bnorm = Bnorm
                        el.alpha = alpha
                        el.Gg = Gg
                        el.rho = new_rho
                        el.drho = new_drho
                        el.b = b
                        el.chi2 = curr_nld.chi2 + ((Gg_mean - el.Gg)/Gg_sigma)**2
                        el.spin_cutoff = new_spincutoff
                        el.D0 = new_D
                    gsfs.append(curr_gsf)
                    if astro and found_astro:
                        ncrates.append(curr_ncrate)
                        
    # save lists of nlds and gsfs to file
    np.save('data/generated/nlds_' + NLD_pathstring + '.npy', nlds)
    np.save('data/generated/gsfs_' + NLD_pathstring + '.npy', gsfs)
    if astro:
        np.save('data/generated/ncrates_' + NLD_pathstring + '.npy', ncrates)

valmatrices = [[],[]]
if astro:
    astrovalmatrix,_ = calc_errors_chis(ncrates)
    header = 'T [GK], best_fit, best_fit-2*sigma, best_fit-sigma, best_fit+sigma, best_fit+2*sigma' 
    np.savetxt('data/generated/ncrates_' + NLD_pathstring + '_whole.txt',astrovalmatrix, header = header)

    MACSvalmatrix = calc_errors_chis_MACS(ncrates)
    header = 'T [GK], best_fit, best_fit-2*sigma, best_fit-sigma, best_fit+sigma, best_fit+2*sigma' 
    np.savetxt('data/generated/MACS_' + NLD_pathstring + '_whole.txt',MACSvalmatrix, header = header)
#Save in best_fits.npy the nld-gsf couple with the least chi2 score
nldchis = []
gsfchis = []
nldvals = []
gsfvals = []

for el in nlds:
    nldchis.append(el.chi2)
    nldvals.append(el.y[energy_bin])
for el in gsfs:
    gsfchis.append(el.chi2)
    gsfvals.append(el.y[energy_bin])
    
nldchi_argmin = np.argmin(nldchis)
gsfchi_argmin = np.argmin(gsfchis)
nldchimin = nldchis[nldchi_argmin]
gsfchimin = gsfchis[gsfchi_argmin]

if astro:
    ncrateschis = []
    ncratesvals = []
    for ncrate in ncrates:
        ncrateschis.append(ncrate.chi2)
        ncratesvals.append(ncrate.ncrate[temp_bin])
    ncrateschi_argmin = np.argmin(ncrateschis)
    ncrateschimin = ncrateschis[ncrateschi_argmin]
    
#save best fits
if astro:
    least_nld_gsf = [nlds[nldchi_argmin], gsfs[gsfchi_argmin], ncrates[ncrateschi_argmin]]
else:
    least_nld_gsf = [nlds[nldchi_argmin], gsfs[gsfchi_argmin]]

for lst, lab, i in zip([nlds, gsfs], ['nld_' + NLD_pathstring,'gsf_' + NLD_pathstring], [0,1]):
    valmatrices[i],_ = calc_errors_chis(lst)
    header = 'Energy [MeV], best_fit, best_fit-2*sigma, best_fit-sigma, best_fit+sigma, best_fit+2*sigma, staterr' 
    writematr = np.c_[valmatrices[i],least_nld_gsf[i].yerr]
    np.savetxt('data/generated/' + lab + '_whole.txt', writematr, header = header)

np.save('data/generated/best_fits_' + NLD_pathstring + '.npy', least_nld_gsf)
    
if plot_chis:
    #plot chi2s for one energy
    fig1, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True)
    if astro:
        fig2, axs2 = plt.subplots(nrows = 1, ncols = 1)
        
    #plot chi distributions
    axs[0].plot(nldvals,nldchis,'b.',alpha=0.4, markersize = 3)
    axs[1].plot(gsfvals,gsfchis,'b.',alpha=0.1, markersize = 3)
    axs[0].plot(nldvals[nldchi_argmin],nldchimin,'k^', label=r'$\chi_{min}^2$')
    axs[1].plot(gsfvals[gsfchi_argmin],gsfchimin,'k^', label=r'$\chi_{min}^2$')
    for i,chimin in enumerate([nldchimin, gsfchimin]):
        axs[i].plot(valmatrices[i][energy_bin,3], chimin+1, 'ro')
        axs[i].plot(valmatrices[i][energy_bin,4], chimin+1, 'ro')
    axs[0].axhline(y=nldchimin+1, color='r', linestyle='--', label=r'$\chi_{min}^2$+1 score')
    axs[1].axhline(y=gsfchimin+1, color='r', linestyle='--', label=r'$\chi_{min}^2$+1 score')
    
    rng = 2
    if astro:
        axs2.plot(ncratesvals,ncrateschis,'b.',alpha=0.5)
        axs2.plot(ncratesvals,ncrateschis,'b.',alpha=1)
        axs2.plot(ncratesvals[ncrateschi_argmin],ncrateschimin,'go', label=r'$\chi_{min}^2$')
        axs2.plot(astrovalmatrix[temp_bin,3], ncrateschimin+1, 'ro')
        axs2.plot(astrovalmatrix[temp_bin,4], ncrateschimin+1, 'ro')
        axs2.axhline(y=ncrateschimin+1, color='r', linestyle='--', label=r'$\chi_{min}^2$+1 score')
        axs2.set_ylim(26.4,40)
    
    #axs[0].set_title(r'$\chi^2$-scores for $\rho(E_x=$ %s MeV)'%"{:.2f}".format(nld.energies[energy_bin]))
    #axs[0].set_ylim(26.4,40)
    #axs[0].set_xlim(180,232)
    axs[0].set_xlabel('NLD [MeV$^{-1}$]')
    axs[0].set_ylabel(r'$\chi^2$-score')
    axs[0].text(0.9, 0.05, 'a)', fontsize='medium', verticalalignment='center', fontfamily='serif', transform = axs[0].transAxes)
    #axs[1].set_title(r'$\chi^2$-scores for $f(E_\gamma=$ %s MeV)'%"{:.2f}".format(gsf.energies[energy_bin]))
    #axs[1].set_ylim(26.4,40)
    #axs[1].set_xlim(7e-9,1.6e-8)
    axs[1].set_xlabel('GSF [MeV$^{-3}$]')
    axs[1].text(0.9, 0.05, 'b)', fontsize='medium', verticalalignment='center', fontfamily='serif', transform = axs[1].transAxes)
    for i in range(rng):
        #axs[i].grid()
        axs[i].legend(loc='upper right', framealpha = 1.)
    if astro:
        axs2.set_xlabel(r'$\sigma_n$ [mb]')
        #axs2.grid()
        axs2.legend()
        axs2.set_title(r'$\chi^2$-scores for $\sigma_n(T=$ %s GK)'%"{:.2f}".format(ncrate.T[temp_bin]))
        fig2.show()
        
    plt.tight_layout()
    fig1.show()
    
    
    
    
    
    
    
