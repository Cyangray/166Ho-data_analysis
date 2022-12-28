#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:11:19 2022

@author: francesco

Script to find out what is the average spin of a nucleus at a certain excitation
energy, given its list of levels.
"""

import numpy as np
import matplotlib.pyplot as plt

level_energies = np.genfromtxt('spin-distr/spins')[:,:-1]
level_spins_str = np.genfromtxt('spin-distr/spins', dtype = str)[:,-1]

def separate_parity_certainty(energies, lss):
    #separate string inputs a la "1+, (6-)" and so on, in a five-column matrix
    #where 1+ = [energy,1,1,1,mult], (6-)=[energy,6,0,0,mult], where the third column is the parity
    #(0 is -, 1 is +), and the fourth column is the certainty (1 is certain, no 
    #parenthesis, while 0 is uncertain, spin in parenthesis), the fifth is how
    #many spins are suggested for the energy level
    output_list = []
    for energy, line in zip(energies[:,0], lss):
        #certain?
        if line[0]=='(':
            certainty = 0
        else:
            certainty = 1
        line = line.replace('(', '')
        line = line.replace(')', '')
        #count number of digits
        ndigits = 0
        for character in line:
            if character.isnumeric():
                ndigits += 1
        for digit in range(ndigits):
            spin_chunk = line[digit*3:digit*3+2]
            spin = float(spin_chunk[0])
            if spin_chunk[1] == '+':
                parity = 1
            else:
                parity = 0
            out_line = [energy, spin, parity, certainty, ndigits]
            output_list.append(out_line)
    return np.array(output_list)

level_spins_whole = separate_parity_certainty(level_energies, level_spins_str)
energy_sigma = 150

def gen_hist_matrix(energy_centroid, energy_sigma, level_spins_whole):
    spin_range = max(level_spins_whole[:,1])
    hist_matrix = np.zeros((int(spin_range + 1), 2))
    hist_matrix[:,0] = np.arange(int(spin_range + 1))
    
    for level in level_spins_whole:
        if (level[0] > (energy_centroid - energy_sigma)) and (level[0] < (energy_centroid + energy_sigma)):
            hist_matrix[int(level[1]),1] += 1/level[4]
    return hist_matrix

centroids = np.arange(150,650,10)
matrices = []
avgs = []
for centroid in centroids:
    matrices.append(gen_hist_matrix(centroid, energy_sigma, level_spins_whole))
    avg = np.sum(matrices[-1][:,0] * matrices[-1][:,1])/np.sum(matrices[-1][:,1])
    avgs.append(avg)

show_mat = np.zeros((len(matrices[0][:,1]), len(matrices)))
for i, matrix in enumerate(matrices):
    show_mat[:,i] = matrix[:,1]
plt.matshow(show_mat, origin = 'lower')
plt.plot(avgs, 'w-')
plt.show()
