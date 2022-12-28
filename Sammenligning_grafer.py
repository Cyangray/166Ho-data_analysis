#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:12:04 2022

@author: francesco
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from systlib import SLO_arglist
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': ['computer Modern'],
    'text.usetex': True,
    'pgf.rcfonts': False,
})

#cmap = matplotlib.cm.get_cmap('YlGnBu')
cmap = matplotlib.cm.get_cmap('tab10')

BSR_constant = 2.5980e8 #u_N^2 MeV^2
E_range1 = np.linspace(2.0,4.0,200)
E_range2 = np.linspace(0,10,500)
b2166Ho = 0.34235 #beta2 deformation of 166Ho, interpolated between 164Dy and 168Er
b2166Ho_moller = 0.296
M_N = 939

#paths
data_path = 'data/generated/'

def nanarray(shape):
    an_array = np.empty(shape)
    an_array[:] = np.NaN
    return an_array

metode1_59_mat = np.loadtxt(data_path + 'metode1_59.txt')
metode1_66_mat = np.loadtxt(data_path + 'metode1_66.txt')
metode2_59_mat = np.loadtxt(data_path + 'metode2_59.txt')
metode2_66_mat = np.loadtxt(data_path + 'metode2_66.txt')
Hildes_metode_2_mat = np.loadtxt(data_path + 'Hildes_metode_2.txt')
Hildes_metode_5_mat = np.loadtxt(data_path + 'Hildes_metode_5.txt')
metoder = [metode1_59_mat, metode1_66_mat, metode2_59_mat, metode2_66_mat]
Hildes_metoder = [Hildes_metode_2_mat, Hildes_metode_5_mat]
metodenr = ['1.6-3.9 MeV', '2.0-3.9 MeV']
metode_label = ['simultaneous fit, T=0.59 GK', 'simultaneous fit, T=0.66 GK', 'GEDR fit first, T=0.59 GK', 'GEDR fit first, T=0.66 GK']
metode_label = ['method 1(A)', 'method 1(B)', 'method 2(A)', 'method 2(B)']

#import data
BSR_renstroem2737 = np.loadtxt('data/nuclear/Dy/BSR_Renstroem2018-2737.txt')
BSR_renstroem0010 = np.loadtxt('data/nuclear/Dy/BSR_Renstroem2018-010.txt')
BSR_Nord2040 = np.loadtxt('data/nuclear/Ho/BSR_Nord2003-2040.txt') #NRF
EGS_renstroem = np.loadtxt('data/nuclear/Dy/EGS_Renstroem2018.txt')
EGS_Siem = np.loadtxt('data/nuclear/Sm/EGS_Siem2002.txt')
EGS_Guttormsen = np.loadtxt('data/nuclear/Nd/EGS_Guttormsen2022.txt')
EGS_Agvaanluvsan_He = np.loadtxt('data/nuclear/Yb/EGS_Agvaanluvsan2004_3He3He.txt')
EGS_Agvaanluvsan_a = np.loadtxt('data/nuclear/Yb/EGS_Agvaanluvsan2004_3Hea.txt')
EGS_Melby = np.loadtxt('data/nuclear/Er/EGS_Melby2001.txt')
EGS_Nord = np.loadtxt('data/nuclear/Ho/EGS_Nord2003.txt') #NRF
EGS_Malatji = np.loadtxt('data/nuclear/Sm/EGS_Malatji2021.txt')
EGS_Simon = np.loadtxt('data/nuclear/Sm/EGS_Simon2016.txt')
EGS_Ziegler = np.loadtxt('data/nuclear/Sm/EGS_Ziegler1990.txt') #NRF

#manually fix some things for readable plots
spacing = 0.1
EGS_Simon[:,0] = [el + spacing for el in EGS_Simon[:,0]]
EGS_Nord[:,0] = [el + spacing for el in EGS_Nord[:,0]]
EGS_Ziegler[:,0] = [el + spacing for el in EGS_Ziegler[:,0]]
EGS_Agvaanluvsan_a[:,0] = [el + spacing for el in EGS_Agvaanluvsan_a[:,0]]
EGS_Siem[:,0] = [el + spacing for el in EGS_Siem[:,0]]
spacing_d = 0.0005

EGS_Simon[:,-1] = [el + spacing_d for el in EGS_Simon[:,-1]]
EGS_Nord[:,-1] = [el + spacing_d for el in EGS_Nord[:,-1]]
EGS_Ziegler[:,-1] = [el + spacing_d for el in EGS_Ziegler[:,-1]]
EGS_Agvaanluvsan_a[:,-1] = [el + spacing_d for el in EGS_Agvaanluvsan_a[:,-1]]

EGSs = [EGS_renstroem, 
        EGS_Siem, 
        EGS_Agvaanluvsan_He, 
        EGS_Melby,
        EGS_Agvaanluvsan_a, 
        EGS_Simon,
        EGS_Guttormsen,
        EGS_Malatji,
        EGS_Nord,
        EGS_Ziegler
        ]

EGSs_labels = ['Dy, Renstrøm et al.',#' 2018', 
               'Sm, Siem et al.',# 2002', 
               r'Yb, Agvaanluvsan et al. ($^3$He,$^3$He)',# 2004 ($^3$He,$^3$He)',
               'Er, Melby et al.',# 2001',
               r'Yb, Agvaanluvsan et al. ($^3$He,$\alpha$)',# 2004 ($^3$He,$\alpha$)', 
               'Sm, Simon et al.',# 2016',
               'Nd, Guttormsen et al.',# 2022',
               'Sm, Malatji et al.',# 2021',
               'Ho, Nord et al.',# 2003',
               'Sm, Ziegler et al.',# 1990'
               ]

def BSR_integral(values, errors, E_range):
        mean = BSR_constant*np.trapz(SLO_arglist(E_range, values),E_range)
        low_err = mean - BSR_constant*np.trapz(SLO_arglist(E_range, np.subtract(values,errors)),E_range)
        upp_err = BSR_constant*np.trapz(SLO_arglist(E_range, np.add(values,errors)),E_range) - mean
        if upp_err > low_err:    
            return [mean, upp_err]
        else:
            return [mean, low_err]

def dfrombeta2(beta2):
    return beta2*np.sqrt(45/(16*np.pi))

def calc_rmi(A, d):
    r0 = 1.15 #fm
    hc = 197.3 #MeV fm
    return 2/5*M_N*r0**2*A**(5/3)*(1+0.31*d)/hc**2

def sum_rule_centroid_BSR(Z, A, d):
    rmi = calc_rmi(A,d)
    omegaD = (31.2*A**(-1/3) + 20.6*A**(-1/6))*(1-0.61*d)
    omegaQ = 64.7*A**(-1/3)*(1-0.3*d)
    xi = omegaQ**2/(omegaQ**2 + 2*omegaD**2)
    centroid = d*omegaD*np.sqrt(2*xi)
    BSR = 3/(4*np.pi)*(Z/A)**2*rmi*d*omegaD*np.sqrt(2*xi)
    return centroid, BSR
    
d = dfrombeta2(b2166Ho_moller)
centroid, BSR = sum_rule_centroid_BSR(67,166,d)

#Plot against the mass number or the deformation? check = True, if mass number.
plot_mass_number = True
#plot average of metode1, 3 and Hildes method?
plot_average_metode = False
plot_average_hildes = True

#fig1, axs1 = plt.subplots(ncols = 3, nrows = 1, figsize=((6.4*3)*0.60,4.8*0.60))
fig1, axs1 = plt.subplots(ncols = 1, nrows = 3, figsize=((5.0*1)*1.1,3.75*3*1.0), dpi = 300)
fig2, axs2 = plt.subplots(ncols = 2, nrows = 2, figsize=((5.0*2)*1.0,(3.75*2)*1.0), dpi=300)
yaxs1_labs = [r'$E_\mathrm{SR}$ (MeV)', r'$\Gamma_\mathrm{SR}$ (MeV)', r'$\sigma_\mathrm{SR}$ (mb)']
yaxs2_labs = [ r'$B_\mathrm{SR}(M1)(\mu_N^2)$ 2.0-4.0 MeV', r'$B_\mathrm{SR}(M1)(\mu_N^2)$ 0-10 MeV']
fig1_labs = ['a)', 'b)', 'c)']
fig2_labs = ['a)', 'b)', 'c)', 'd)']
empty_circle = matplotlib.markers.MarkerStyle(marker='o', fillstyle='none')

# FIG 1
if plot_mass_number:
    x_data_col = 0
    x_label = 'A'
    x_lim = [141.5, 173.5]
    xname = 'VS-A-V'
else:
    x_data_col = 11
    x_label = r'$\beta_2$'
    x_lim = [0.06, 0.40]
    xname = 'VS-beta'
for i, ax in enumerate(axs1):
    ax.text(0.9, 0.05, fig1_labs[i], fontsize='large', verticalalignment='center', fontfamily='serif', transform = ax.transAxes)
    for j, metode in enumerate(metoder):
        if plot_mass_number:
            x1 = metode[0]+(j*spacing)
            x2 = metode[0]+(j*spacing + spacing*2)
            x3 = metode[0]+(j*spacing + spacing*3)
            x4 = metode[0]+(j*spacing + spacing*4)
        else:
            x1 = x2 = x3 = x4 = metode[11]
        xs = [x1, x2, x3, x4]
        ax.errorbar(xs[j], metode[2*i+1], metode[2*i+2], fmt = 'v', mfc='none', label = 'Ho, this work, ' + metode_label[j], color = cmap(j/5))
    for j, EGS in enumerate(EGSs):
        marker = 'o'
        if j in [8, 9]:
            marker = 'x'
        if 0 not in EGS[:,2*i+1]:
            ax.errorbar(EGS[:,x_data_col], EGS[:,2*i+1], EGS[:,2*i+2], fmt=marker, mfc = 'none', color = cmap(j/10), label = EGSs_labels[j])
    ax.set_xlim(x_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(yaxs1_labs[i])
handles, labels = axs1[0].get_legend_handles_labels()
lgd = fig1.legend(handles, labels, bbox_to_anchor = (0.5, -0.13), ncol= 2, loc = 'lower center')
fig1.tight_layout()
fig1.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig1.axes[:-1]], visible=False)
fig1.savefig('pictures/3-graphs' + xname + '.pdf', bbox_extra_artists=(lgd, ), bbox_inches='tight', format = 'pdf')
fig1.show()


#FIG 2

axs2f = axs2.flatten()
#Main subplots
for i, ax in enumerate(axs2f):
    if i in [0,2]:
        plot_mass_number = True
        x_data_col = 0
        x_label = 'A'
        x_lim = [141.5, 173.5]
    else:
        plot_mass_number = False
        x_data_col = 11
        x_label = r'$\beta_2$'
        x_lim = [0.06, 0.36]
        
    if i in [0,1]:
        yax_lab = yaxs2_labs[0]
    else:
        yax_lab = yaxs2_labs[1]
    ax.text(0.92, 0.05, fig2_labs[i], fontsize='large', verticalalignment='center', fontfamily='serif', transform = ax.transAxes)
    if i in [0,1]:
        for j, metode in enumerate(metoder):
            
            if plot_mass_number:
                x1 = metode[0]+(j*spacing)
                x2 = metode[0]+(j*spacing + spacing*2)
                x3 = metode[0]+(j*spacing + spacing*3)
                x4 = metode[0]+(j*spacing + spacing*4)
            else:
                x1 = b2166Ho - spacing_d
                x2 = b2166Ho
                x3 = b2166Ho + spacing_d
                x4 = b2166Ho + spacing_d *2
            xs = [x1,x2,x3,x4]
            ax.errorbar(xs[j], metode[7], metode[8], fmt = 'v', mfc='none', label = 'Ho, this work, ' + metode_label[j], color = cmap(j/5))
        if plot_average_hildes:
            max_up = max([Hildes_metoder[0][0,i] + Hildes_metoder[0][1,i], Hildes_metoder[1][0,i] + Hildes_metoder[1][1,i]])
            min_down = min([Hildes_metoder[0][0,i] - Hildes_metoder[0][1,i], Hildes_metoder[1][0,i] - Hildes_metoder[1][1,i]])
            average = (max_up + min_down)/2
            unc = max_up - average
            ax.errorbar(x1, average, unc, fmt = 'v', mfc='none', color = cmap(4/5), label = 'Ho, this work, method 3')# label = 'Ho, exp. background')
        else:
            for j, metode in enumerate(Hildes_metoder):
                if plot_mass_number:
                    x1 = 166 - (j+1)*spacing
                else:
                    x1 = b2166Ho
                ax.errorbar(x1, metode[0,i], metode[1,i], fmt = 'v', mfc='none', label = 'Ho, exp. background, SR in ' + metodenr[j])
        for j, EGS in enumerate(EGSs):
            marker = 'o'
            if j in [8, 9]:
                marker = 'x'
            if 0 not in EGS[:,7]:
                ax.errorbar(EGS[:,x_data_col], EGS[:,7], EGS[:,8], fmt=marker, label = EGSs_labels[j], mfc = 'none', color = cmap(j/10))
            else:
                curr_BSR = np.zeros((EGS.shape[0], 2))
                for k in range(EGS.shape[0]):
                    mean, err = BSR_integral([EGS[k,1], EGS[k,3], EGS[k,5]], [EGS[k,2], EGS[k,4], EGS[k,6]], E_range1)
                    curr_BSR[k,:] = [mean,err]
                ax.errorbar(EGS[:,x_data_col], curr_BSR[:,0], curr_BSR[:,1], fmt=marker, mfc = 'none', label = EGSs_labels[j], color = cmap(j/10))
                
    elif i in [2,3]:
        for j, metode in enumerate(metoder):
            if plot_mass_number:
                x1 = metode[0]+(j*spacing)
                x2 = metode[0]+(j*spacing + spacing*2)
                x3 = metode[0]+(j*spacing + spacing*3)
                x4 = metode[0]+(j*spacing + spacing*4)
            else:
                x1 = b2166Ho - spacing_d
                x2 = b2166Ho
                x3 = b2166Ho + spacing_d
                x4 = b2166Ho + spacing_d *2
            xs = [x1,x2,x3,x4]
            ax.errorbar(xs[j], metode[9], metode[10], fmt = 'v', mfc='none', label = 'Ho, this work, ' + metode_label[j], color = cmap(j/5))
        
        if plot_average_hildes:
            max_up = max([Hildes_metoder[0][0,i-2] + Hildes_metoder[0][1,i-2], Hildes_metoder[1][0,i-2] + Hildes_metoder[1][1,i-2]])
            min_down = min([Hildes_metoder[0][0,i-2] - Hildes_metoder[0][1,i-2], Hildes_metoder[1][0,i-2] - Hildes_metoder[1][1,i-2]])
            average = (max_up + min_down)/2
            unc = max_up - average
            ax.errorbar(x1, average, unc, fmt = 'v', mfc='none', color = cmap(4/5), label = 'Ho, Hilde\'s method')
        else:
            for j, metode in enumerate(Hildes_metoder):
                if plot_mass_number:
                    x1 = 166 - (j+1)*spacing
                else:
                    x1 = b2166Ho - spacing_d*2
                ax.errorbar(x1, metode[0,i-2], metode[1,i-2], fmt = 'v', mfc='none',  label = 'Ho, Hilde\'s method, SR in ' + metodenr[j])
        
        for j, EGS in enumerate(EGSs):
            marker = 'o'
            if j in [8, 9]:
                marker = 'x'
            if 0 not in EGS[:,9]:
                ax.errorbar(EGS[:,x_data_col], EGS[:,9], EGS[:,10], fmt=marker, mfc = 'none', color = cmap(j/10), label = EGSs_labels[j])
            else:
                curr_BSR = np.zeros((EGS.shape[0], 2))
                for k in range(EGS.shape[0]):
                    mean, err = BSR_integral([EGS[k,1], EGS[k,3], EGS[k,5]], [EGS[k,2], EGS[k,4], EGS[k,6]], E_range2)
                    curr_BSR[k,:] = [mean,err]
                ax.errorbar(EGS[:,x_data_col], curr_BSR[:,0], curr_BSR[:,1], fmt=marker, mfc = 'none', color = cmap(j/10), label = EGSs_labels[j])
    ax.set_xlim(x_lim)

handles, labels = axs2[0,0].get_legend_handles_labels()
lgd = fig2.legend(handles, labels, bbox_to_anchor = (0.5, -0.135), ncol= 4, loc = 'lower center')
fig2.tight_layout()
fig2.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig2.axes[:-2]], visible=False)
fig2.subplots_adjust(wspace=0)
axs2f[2].set_xlabel('A')
axs2f[3].set_xlabel(r'$\beta_2$')
axs2f[0].set_ylabel(yaxs2_labs[0])
axs2f[2].set_ylabel(yaxs2_labs[1])
plt.setp(fig2.axes[1].get_yticklabels(), visible=False)
plt.setp(fig2.axes[3].get_yticklabels(), visible=False)

#insets
axins1 = inset_axes(axs2f[1], width=2.3, height=1.7, loc=2, borderpad = 2)
axins2 = inset_axes(axs2f[3], width=2.3, height=1.7, loc=2, borderpad = 2)
axins = np.array([axins1, axins2])
for ins_i, ax in enumerate(axins):
    i = ins_i*2 + 1

    plot_mass_number = False
    x_data_col = 11
    x_label = r'$\beta_2$'
    x_lim = [0.32, 0.351]
        
    if ins_i == 0:
        yax_lab = yaxs2_labs[0]
        y_lim = [0.5,6]
    else:
        yax_lab = yaxs2_labs[1]
        y_lim = [0.5,8]
    
    if ins_i == 0:
        for j, metode in enumerate(metoder):
            
            if plot_mass_number:
                x1 = metode[0]+(j*spacing)
                x2 = metode[0]+(j*spacing + spacing*2)
                x3 = metode[0]+(j*spacing + spacing*3)
                x4 = metode[0]+(j*spacing + spacing*4)
            else:
                x1 = b2166Ho - spacing_d
                x2 = b2166Ho
                x3 = b2166Ho + spacing_d
                x4 = b2166Ho + spacing_d *2
            xs = [x1,x2,x3,x4]
            ax.errorbar(xs[j], metode[7], metode[8], fmt = 'v', mfc='none', color = cmap(j/5), label = 'Ho, this work, ' + metode_label[j])
            
        if plot_average_hildes:
            max_up = max([Hildes_metoder[0][0,i] + Hildes_metoder[0][1,i], Hildes_metoder[1][0,i] + Hildes_metoder[1][1,i]])
            min_down = min([Hildes_metoder[0][0,i] - Hildes_metoder[0][1,i], Hildes_metoder[1][0,i] - Hildes_metoder[1][1,i]])
            average = (max_up + min_down)/2
            unc = max_up - average
            ax.errorbar(x1 - spacing_d, average, unc, fmt = 'v', mfc='none', color = cmap(4/5), label = 'Ho, this work, method 3')
        else:
            for j, metode in enumerate(Hildes_metoder):
                if plot_mass_number:
                    x1 = 166 - (j+1)*spacing
                else:
                    x1 = b2166Ho
                ax.errorbar(x1, metode[0,i], metode[1,i], fmt = 'v', mfc='none', label = 'Ho, exp. background, SR in ' + metodenr[j])
        for j, EGS in enumerate(EGSs):
            marker = 'o'
            if j in [8, 9]:
                marker = 'x'
            if 0 not in EGS[:,7]:
                ax.errorbar(EGS[:,x_data_col], EGS[:,7], EGS[:,8], fmt=marker, mfc = 'none', color = cmap(j/10), label = EGSs_labels[j])
            else:
                curr_BSR = np.zeros((EGS.shape[0], 2))
                for k in range(EGS.shape[0]):
                    mean, err = BSR_integral([EGS[k,1], EGS[k,3], EGS[k,5]], [EGS[k,2], EGS[k,4], EGS[k,6]], E_range1)
                    curr_BSR[k,:] = [mean,err]
                ax.errorbar(EGS[:,x_data_col], curr_BSR[:,0], curr_BSR[:,1], fmt=marker, mfc = 'none', color = cmap(j/10), label = EGSs_labels[j])
        
    elif ins_i == 1:
        for j, metode in enumerate(metoder):
            if plot_mass_number:
                x1 = metode[0]+(j*spacing)
                x2 = metode[0]+(j*spacing + spacing*2)
                x3 = metode[0]+(j*spacing + spacing*3)
                x4 = metode[0]+(j*spacing + spacing*4)
            else:
                x1 = b2166Ho - spacing_d
                x2 = b2166Ho
                x3 = b2166Ho + spacing_d
                x4 = b2166Ho + spacing_d *2
            xs = [x1,x2,x3,x4]
            ax.errorbar(xs[j], metode[9], metode[10], fmt = 'v', mfc='none', color = cmap(j/5), label = 'Ho, ' + metode_label[j])
        
        if plot_average_hildes:
            max_up = max([Hildes_metoder[0][0,i-2] + Hildes_metoder[0][1,i-2], Hildes_metoder[1][0,i-2] + Hildes_metoder[1][1,i-2]])
            min_down = min([Hildes_metoder[0][0,i-2] - Hildes_metoder[0][1,i-2], Hildes_metoder[1][0,i-2] - Hildes_metoder[1][1,i-2]])
            average = (max_up + min_down)/2
            unc = max_up - average
            ax.errorbar(x1 - spacing_d, average, unc, fmt = 'v', mfc='none', color = cmap(4/5), label = 'Ho, Hilde\'s method')
        else:
            for j, metode in enumerate(Hildes_metoder):
                if plot_mass_number:
                    x1 = 166 - (j+1)*spacing
                else:
                    x1 = b2166Ho - spacing_d*2
                ax.errorbar(x1, metode[0,i-2], metode[1,i-2], fmt = 'v', mfc='none', label = 'Ho, Hilde\'s method, SR in ' + metodenr[j])
        
        for j, EGS in enumerate(EGSs):
            marker = 'o'
            if j in [8, 9]:
                marker = 'x'
            if 0 not in EGS[:,9]:
                ax.errorbar(EGS[:,x_data_col], EGS[:,9], EGS[:,10], fmt=marker, label = EGSs_labels[j], color = cmap(j/10), mfc = 'none')
            else:
                curr_BSR = np.zeros((EGS.shape[0], 2))
                for k in range(EGS.shape[0]):
                    mean, err = BSR_integral([EGS[k,1], EGS[k,3], EGS[k,5]], [EGS[k,2], EGS[k,4], EGS[k,6]], E_range2)
                    curr_BSR[k,:] = [mean,err]
                ax.errorbar(EGS[:,x_data_col], curr_BSR[:,0], curr_BSR[:,1], fmt=marker, color = cmap(j/10), mfc = 'none', label = EGSs_labels[j])
    ax.set_xlim(x_lim)    
    ax.set_ylim(y_lim)

fig2.savefig('pictures/2x2-graphs.pdf', bbox_extra_artists=(lgd, ), bbox_inches='tight', format = 'pdf')
fig2.show()