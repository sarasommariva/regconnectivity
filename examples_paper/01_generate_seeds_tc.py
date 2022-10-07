#!/usr/bin/env python
# coding: utf-8
"""
Part 1: Generate seeds time coursed and locations
=================================================

"""

import mne
import numpy as np
import math
import os.path as op
from scipy import signal
import functions_code as myf
import sys
target  = '..'
noct = '6'


#################################################################################
# Loading data

# data path
file_fwd = op.join(target,'data','oct'+noct+'_fwd.fif')

# load fwd
fwd = mne.read_forward_solution(file_fwd, verbose=False)
fwd = mne.convert_forward_solution(fwd, surf_ori=True,force_fixed=True, use_cps=True, verbose=False)
fwd = mne.pick_types_forward(fwd, meg='mag', eeg=False, ref_meg=False)

G = fwd['sol']['data'] # leadfield matrix
G = 10**5*G # rescale to avoid small numbers


# dipole position
dip_pos = fwd['source_rr']

# dipole orientations
dip_or = fwd['source_nn']

# load cortico-cortical distance matrix
cortico_dist_file = op.join(target,'data','cortico_dist_oct'+noct+'.npy')
cortico_dist = np.load(cortico_dist_file)

#################################################################################
# define features

N_mod = int(sys.argv[1])  # Number of simulated AR models (with connections)
N_loc = int(sys.argv[2])  # Number of different connected pairs of locations
alpha = np.random.rand(N_mod)*0.9+0.1 # Standard deviation of the entries of the AR model
alpha.sort()

T = int(10000)   # Number of time points
fs = int(128)    # Samples frequency
delta_t = 1/fs
fmin = 8
fmax = 12

M = G.shape[0] # Number of sensor
N_dense = G.shape[1] # Number of sources in the dense source space
N_act = int(2)  # Number of active patches
P = int(5)   # MVAR order
ratio_max = 1.5 # Ratio max between the intensities of the seed sources

# store relevant features
features = {'N_mod':N_mod, 'N_loc':N_loc, 'T':T, 'fs':fs, 'fmin':fmin, 'fmax':fmax, 'N_act':N_act}

#################################################################################
# generate N_mod MVAR models of dimension 2 (pairs of time courses)

# Initialize the time courses of the seeds of the active patches
seed_tcs = np.zeros((N_act,T,N_mod))

nperseg = 256 # length of the window for the fourier transform
nfft = nperseg # number of frequencies
i_mod = 0
while i_mod <N_mod:
    ratio = np.inf
    while ratio > ratio_max:
        AR_mod = myf.gen_ar_model(N_act, P, alpha[i_mod])
        X_act, AR_mod = myf.gen_ar_series(AR_mod, T)
        norm_X = np.linalg.norm(X_act, axis = 1)
        norm_X.sort()
        ratio = norm_X[-1]/norm_X[0] # retain pairs of time courses whose intensities are close (int_max/int_min < ratio_max)
    norm_const = np.mean(np.std(X_act, axis = 1))
    X_act = X_act/norm_const # normalize the time series so that different pairs have similar intensity
    # compute power spectrum in the freq range of interest
    f, Pwe = signal.welch(X_act, fs=fs, window='hann', nperseg=nperseg,
                          noverlap=nperseg // 2, nfft=nfft, detrend='constant',
                          return_onesided=True, scaling='density', axis=-1)
    P_tot = Pwe[0,:]+Pwe[1,:]
    f_in = np.intersect1d(np.where(f>=fmin)[0],np.where(f<=fmax)[0])
    # retain only time courses with sufficiently high power in the frequency range of interest
    if np.sum(P_tot[f_in])/len(f_in)>1.2*np.sum(P_tot)/len(f):
        b,a = signal.butter(3, np.array([8,12]), btype='bandpass', analog=False, output='ba', fs=fs)
        X_act = signal.filtfilt(b, a, X_act, axis=- 1, padtype='odd', padlen=None, method='pad', irlen=None)
        seed_tcs[:,:,i_mod] = X_act
        i_mod = i_mod +1
     
    

#################################################################################
# generate patches sources

# select pairs of active sources (seeds) in the source space 
seed_locs = myf.select_sources(G,dip_pos,N_loc) 


#################################################################################
# store the generated data in a dictionary and save it

seed_tc_loc={'features':features, 'seed_tcs':seed_tcs, 'seed_locs':seed_locs}
np.save('./run_data/seed_tc_loc.npy', seed_tc_loc)





