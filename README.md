# 166Ho-data_analysis
Data analysis files for the 163Dy(a,p)166Ho experiment

Paper in publication process. Link and DOI will be added when published.

Similarly to Systlib, this is firstly a backup of the code I used to analyze data from the 163Dy(alpha,p gamma)166Ho experiment run at OLC in 2018, and secondly a reference for future experiments. This set of scripts are meant as a way to propagate the uncertainties in the low excitation energy level density, the spin-cutoff parameter, D0 and <Gg> to the final nuclear level density and gamma strength function for 166Ho evaluated with the Oslo method. In brief, this set of scripts expands on the oslo-method-software package.
The procedure, broadly speaking, calculates the systematic errors by "brute force". It runs counting.c and normalization.c many times in order to generate many different nlds and gsfs using different parameters as input (here we loop though different L1, L2, rho(Sn) connected to different choices of spin-cutoff parameter, and <Gg>.
For each nld and gsf, it evaluates their fit to:
1) known levels in a region in the low excitation energy, 
2) chosen value(s) of rho(Sn) with uncertainty, 
3) chosen value(s) of <Gg> with uncertainty guessed by calculating a chi2 score.
It then generates input for talys to calculate the neutron capture rate and MACS and evaluates the uncertainties for the nld, gsf, ncrate and MACS by finding graphically where the chi2_min + 1 intercepts the parabola in the nld (or gsf or MACS) vs chi2 plot.

Generally speaking, this is what I did:
1) Run run_cnt_nrm.py. This script creates a database by looping counting.c and normalization.c several times for different parameters. L1 and L2 are chosen to cover all possible interval choices for the low energy region, and rho(Sn) and to cover the uncertainty span evaluated in the step above.
2) create_talys_inputs.py creates a set of inputs for talys by creating a Ho.tab (the way talys 1.96/2.00 reads the level density), an input file and E1strength and M1 strength files in the same folder structure as the dataset. The E1 and M1 strength are extrapolated for higher energies by joining the gsf to the fit to the experimental GDR from Varlamov et al. for 165Ho. This is then provided to Saga, which will run through them. I then join the output with the dataset of the nlds and gsfs, so that each folder now includes the evaluated neutron capture rates and MACS.
3) run make_nld_gsfs_list.py. Here we evaluate the chi2 score of each generated nld, gsf and astrorate, and collect all properties of each evaluated nld, gsf and astrorate in objects, which are successively collected in numpy arrays, saved to file. Here we also evaluate graphically the standard deviation for each energy or temperature bin by checking where the chi2_min line crosses the parabola in the value vs chi2 graph. The standard deviation values are then written to txt files, easily readable for other codes.
4) create_talys_inputs_stats.py runs talys for four different scenarios corresponding to the lower and upper errors for the best fitting nld and gsf, thus propagating the statistical and systematic experimental errors to the astrophysical quantities (neutron-capture rate and MACS)
5) plot_nld_gsf_errs.py takes the txt files and plots them together with models from talys.
6) plot_165Ho_MACS.py plots the MACS together with the values from different libraries and TALYS.

Together with these files, there are:
1) chi2-test-FG-CT.py runs a fitting algorithm in order to find out if the Fermi gas or the costant temperature model fits best to the experimental data. Used to choose which extrapolation to use in counting.c in the oslo-method-software package.
2) spin_distributions.py is used to find out the average spin of the nucleus at a certain excitation energy
3) BSR_from_exp.py calculates and writes to file the area underneath the scissor resonance by modeling the E1 strength underneath as a simple exponential. Corresponds to method 3 in the article.
4) Metode1.py Fits the GSF with simpler functions/structures, and writes the parameters to file. Corresponds to method 1 in the paper.
5) Metode2.py Fits the GSF with simpler functions/structures, by first fitting the GDR, then the scissors resonance and the PDR, and writes the parameters to file. Corresponds to method 2 in the paper.
6) Sammenligning_grafer.py plots the parameters found in the three previous scripts, and compares to other data from literature.

NB: in order to run these scripts, some packages have to be installed via pip, such as the usual numpy, matplotlib but also iminuit. Check the header of the scripts to find out.
A modified version of the counting.c and normalization.c from oslo-method-software were used, so that these could be automatised in Python. The only changes are that all input and output lines are commented out, so that the only input comes from the input.cnt and input.nrm files that are created atomatically in run_cnt_nrm.py.
