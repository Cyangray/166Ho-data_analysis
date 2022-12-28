#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Aug 20 12:17:28 2022

@author: francesco, last update December 28th 2022

Script to loop through counting.c and normalization.c from the Oslo method.
It creates a directory structure with different input.cnt with different values
of L1, L2 and spin-cutoff parameter (thus, rho(Sn)), and runs counting on them. 
The results are written in each itsfolder. 
Then for each of these, normalization.c is run with different values of <Gamma_gamma>
Preparations: 
1) in the same folder as this script, put copies of counting.dat, 
rhosp.rsg, sigsp.rsg, rhopaw.rsg, sigpaw.rsg files in the same folder
2) run counting once as you normally would. Copy the values in input.cnt in the 
variable "newinput_cnt", changing the values for L1, L2 and rho with the corresponding 
code variables. Use the example for 166Ho here as an example. The spin-cutoff parameter
is varied and the distribution is the one used in the "ALEX" method in counting.c.
The values of low Ex and average spin at that value (the values 0.200000 4.500000
in the input.cnt file) are calculated with the spin_distribution.py script.
3) similarly for normalization: run it once as you normally would, and change the
values in newinput_nrm with the one you got, keeping the name of the python variables
here.

NB: it needs a slightly modified (and recompiled) counting.c and normalization, 
where all lines where it asks input (sscanf, I think) must be commented out, so 
that it only imports data from input.cnt and input.nrm.
to make the program run smoother, comment out as well all printed messages in
the c file.
Change the paths so that it reflects the folder setup on your computer.
'''

from subprocess import call
import os
import numpy as np

NLD_pathstring = 'FG'
    
#master folder name:
master_folder = '/home/francesco/Documents/163Dy-alpha-2018/Fittings/Make_dataset/166Ho-database_' + NLD_pathstring

#full adress of modified counting and normalization codes
counting_code_path = "/home/francesco/oslo-method-software-auto/prog/counting"
normalization_code_path = "/home/francesco/oslo-method-software-auto/prog/normalization"

def rho2D(rho, target_spin, spin_cutoff):
    '''
    calculate D from rho at Bn (Larsen et al. 2011, eq. (20))
    Takes as input rho as 1/MeV, and gives output D as eV
    target_spin is the spin of the target nucleus
    spin_cutoff is self-explanatory - calculate with robin (?)
    '''
    factor = 2*spin_cutoff**2/((target_spin + 1)*np.exp(-(target_spin + 1)**2/(2*spin_cutoff**2)) + target_spin*np.exp(-target_spin**2/(2*spin_cutoff**2)))
    D0 = factor/rho
    return D0*1e6

def D2rho(D0, target_spin, spin_cutoff):
    '''
    calculate D from rho at Bn (Larsen et al. 2011, eq. (20))
    Takes as input rho as 1/MeV, and gives output D as eV
    target_spin is the spin of the target nucleus
    spin_cutoff is self-explanatory - calculate with robin (?)
    '''
    factor = 2*spin_cutoff**2/((target_spin + 1)*np.exp(-(target_spin + 1)**2/(2*spin_cutoff**2)) + target_spin*np.exp(-target_spin**2/(2*spin_cutoff**2)))
    rho = factor/(D0*1e-6)
    return rho

def drho(target_spin, sig, dsig, D0, dD0, rho = None):
    '''
    Calculate the uncertainty in rho, given the input parameters. sig and dsig
    are the spin cutoff parameter and its uncertainty, respectively. Code taken
    from D2rho in the oslo_method_software package.
    '''
    
    alpha = 2*sig**2
    dalpha = 4*sig*dsig
    if target_spin == 0:
        y1a = (target_spin+1.0)*np.exp(-(target_spin+1.0)**2/alpha)
        y1b = (target_spin+1.)**2*y1a
        z1  = y1b
        z2  = y1a
    else:
        y1a = target_spin*np.exp(-target_spin**2/alpha)
        y2a = (target_spin+1.)*np.exp(-(target_spin+1.)**2/alpha)
        y1b = target_spin**2*y1a
        y2b = (target_spin+1.)**2*y2a
        z1  = y1b+y2b
        z2  = y1a+y2a
    u1 = dD0/D0
    u2 = dalpha/alpha
    u3 = 1-z1/(alpha*z2)
    if rho == None:
        rho = D2rho(D0, target_spin, sig)
    return rho*np.sqrt(u1**2 + u2**2*u3**2)

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
spin_cutoff_low = 5.546 #G&C
spin_cutoff_high = 6.926 #RMI
cutoff_unc = 0.00
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

try:
    os.mkdir(master_folder)
except:
    pass
os.chdir(master_folder)
for b in blist:
    bstr_int = "{:.2f}".format(b)
    bstr = bstr_int.translate(bstr_int.maketrans('','', '.'))
    new_spincutoff = b*base_spin_cutoff
    new_spincutoff_str = '{:.6f}'.format(new_spincutoff)
    new_rho = D2rho(D0, target_spin, new_spincutoff)
    new_drho = drho(target_spin, new_spincutoff, new_spincutoff*cutoff_unc, D0, D0_err, rho = new_rho)
    new_D = D0
    new_rho_str = '{:.6f}'.format(new_rho)
    new_drho_str = '{:.6f}'.format(new_drho)
    new_D_str = '{:.6f}'.format(new_D)
    new_dir_rho = bstr + '-' + str(int(new_rho))
    try:
        os.mkdir(new_dir_rho)
    except:
        pass
    os.chdir(new_dir_rho) 
    for L1n in range(L1min,L1max):
        L1 = str(L1n)
        if L1n == 1:
            L2_skip = 2
        else:
            L2_skip = 1
        for L2n in range(L1n + L2_skip, L1max):
            L2 = str(L2n)
            new_dir_L1_L2 = 'L1-'+L1+'_L2-'+L2
            try:
                os.mkdir(new_dir_L1_L2)
            except:
                pass
            os.chdir(new_dir_L1_L2)
            os.system('cp ../../../counting.dat counting.dat')
            os.system('cp ../../../rhosp.rsg rhosp.rsg')
            os.system('cp ../../../sigsp.rsg sigsp.rsg')
            os.system('cp ../../../rhopaw.rsg rhopaw.rsg')
            os.system('cp ../../../sigpaw.rsg sigpaw.rsg')
            newinput_cnt = ' 166.000000 1.847000 6.243640 ' + new_rho_str +' ' + new_drho_str + ' \n ' + L1 + ' ' + L2 + ' 32 45 \n 18 19 41 45 \n 5 18.280001 -0.949000 \n 2 0.562000 -1.884088 \n 2 \n 0 -1000.000000 -1000.000000 \n 0 -1000.000000 -1000.000000 \n 1.000000 \n 0.200000 4.500000 6.243640 ' + new_spincutoff_str + ' \n 150.000000 '
            with open('input.cnt', 'w') as write_obj:
                write_obj.write(newinput_cnt)
            
            call([counting_code_path])
            
            for Gg in Gglist:
                Ggstr = str(int(Gg))
                Gg_input_str = '{:.6f}'.format(Gg)
                try:
                    os.mkdir(Ggstr)
                except:
                    pass
                os.chdir(Ggstr)
                os.system('cp ../rhosp.rsg rhosp.rsg')
                os.system('cp ../rhotmopaw.cnt rhotmopaw.cnt')
                os.system('cp ../sigextpaw.cnt sigextpaw.cnt')
                os.system('cp ../spincut.cnt spincut.cnt')
                os.system('cp ../sigpaw.cnt sigpaw.cnt')
                os.system('cp ../input.cnt input.cnt')
                newinput_nrm = ' 0 6.243000 3.500000 \n ' + new_D_str + ' ' + Gg_input_str + ' \n 105.000000 150.000000 '
                with open('input.nrm', 'w') as write_obj:
                    write_obj.write(newinput_nrm)
                call([normalization_code_path]);
                os.system('rm rhosp.rsg')
                os.system('rm rhotmopaw.cnt')
                os.system('rm sigextpaw.cnt')
                os.system('rm spincut.cnt')
                os.system('rm sigpaw.cnt')
                os.system('rm input.cnt')
                os.chdir('..')
                
            os.system('rm counting.dat')
            os.system('rm rhosp.rsg')
            os.system('rm sigsp.rsg')
            os.system('rm rhopaw.rsg')
            os.system('rm sigpaw.rsg')
            os.chdir('..')
    print('bstr: ' + bstr)
    os.chdir('..')
