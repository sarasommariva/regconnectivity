"""
Single run
==========

Simulate sensor level MEG configuration and compute optimal parameters for neural activity and connectivity estimation
"""

#!/usr/bin/env python
# coding: utf-8

# # 

# In[1]:


import mne
import numpy as np
import math
import os.path as op
import os
from scipy import optimize, signal
import functions_code as myf
from mne import (read_forward_solution, convert_forward_solution, 
                 pick_types_forward)

mne.viz.set_3d_backend("notebook")
get_ipython().run_line_magic('matplotlib', 'notebook')

target  = '..'
i_sim = '8'
subject_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
subject = 'sample'

hemi = 'both' # for visualization


# ## Define customizable features

# In[2]:


alpha = np.random.rand(1)*0.9+0.1 # Standard deviation of the entries of the AR model (accepted values are: raning from 0.1 to 1)
area = 8 # area of the simulated patch in cm^2 (accepted values are: 2, 4, 8) 
intra_coh = 1 # intra coherenc of the simulated patch (accepted values are: 1, 2, 4)
SNR_sensors = -5 # sensor level SNR in dB (accepted values are: ranging from -20 to 5)
SNR_background = 2 # background activity (computed as the norm f the patches time cources over the norm of the background time courses)
                   # (accepted values are: 0.1, 0.5, 0.9)


# # Loading data

# In[3]:


# data path
noct = '6'  # density of the source space
file_fwd = op.join(target,'data','oct'+noct+'_fwd.fif')

# load data
fwd = mne.read_forward_solution(file_fwd, verbose=False)
fwd = mne.convert_forward_solution(fwd, surf_ori=True,force_fixed=True, use_cps=True, verbose=False)
fwd = mne.pick_types_forward(fwd, meg='mag', eeg=False, ref_meg=False)

G = fwd['sol']['data'] # leadfield matrix
G = 10**5*G
GGt = G.dot(G.T) 
U, s, V = np.linalg.svd(G)
V = V.T

# dipols position
dip_pos = fwd['source_rr']

# dipols orientations
dip_or = fwd['source_nn']

# load cortico-cortical distance matrix
cortico_dist_file = op.join(target,'data','cortico_dist_oct'+noct+'.npy')
cortico_dist = np.load(cortico_dist_file)

# Source space
src = fwd['src']

# vertex indeces
vertno = [src[0]['vertno'], src[1]['vertno']]


# ## Define additional features

# In[4]:


T = int(10000)   # Number of time points
fs = int(128)    # Samples frequency

# range for filtering the data
fmin = 8 
fmax = 12

delta_t = 1/fs # Time resolution
M = G.shape[0] # Number of sensor
N_dense = G.shape[1] # Number of sources in source space
N_act = int(2)  # Number of active patches
P = int(5)   # MVAR order
ratio_max = 1.5 # Ratio max between the intensities of the seed sources of the active patches

# store relevant features
features = {'T':T,
            'fs':fs,
            'fmin':fmin,
            'fmax':fmax,
            'N_act':N_act,
            'SNR_sensors':SNR_sensors,
            'area':area,
            'intra_coh':intra_coh,
            'SNR_backgroud':SNR_background
           }


# ## Define sources location

# In[5]:


# Select a pais of sources satisfying specific requirements
seed_loc = myf.select_sources(G,dip_pos,1)[0] 

# define patch radius from the desired area
r = np.sqrt(area*10**(-4)/math.pi) # radius of the patch (maximum distance from the seed in meters)

# define the location of the sources within the patches
p1_locs, p2_locs = myf.gen_patches_sources(cortico_dist, r, seed_loc)


# # Plot patches

# In[6]:


X = np.zeros((G.shape[1]))
stc = mne.SourceEstimate(X, vertices=vertno, tmin=0, tstep=1, subject=subject)

brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=False, colorbar=True)

nv_lh = stc.vertices[0].shape[0]
for idx, loc in enumerate([loc for loc in p1_locs]):
    if loc < nv_lh:
        brain.add_foci(stc.vertices[0][loc], coords_as_verts=True,
                       hemi='lh', color='black', scale_factor=0.3)
    else:
        brain.add_foci(stc.vertices[1][loc-nv_lh], coords_as_verts=True,
                       hemi='rh', color='black', scale_factor=0.3)
        
nv_lh = stc.vertices[0].shape[0]
for idx, loc in enumerate([loc for loc in p2_locs]):
    if loc < nv_lh:
        brain.add_foci(stc.vertices[0][loc], coords_as_verts=True,
                       hemi='lh', color='red', scale_factor=0.3)
    else:
        brain.add_foci(stc.vertices[1][loc-nv_lh], coords_as_verts=True,
                       hemi='rh', color='red', scale_factor=0.3)


# # Simulate brain activity

# ## Step 1: simulate a pair of MVAR time courses
# 
# The generated time courses must satisfy two criteria:
# 1) the ratio between the norms is lower than ratio_max
# 
# 2) the power spectrum in the frequency range of interest is high enough

# In[7]:


# initialize the time courses
seed_tc = np.zeros((N_act,T))

nperseg = 256 # length of the window for the fourier transform
nfft = nperseg # number of frequencies

power_condition = 0
while power_condition == 0:
    ratio = np.inf
    
    # generate MVAR time courses until they meet the condition on the norm
    while ratio > ratio_max:
        AR_mod = myf.gen_ar_model(N_act, P, alpha) # generate MVAR model
        X_act, AR_mod = myf.gen_ar_series(AR_mod, T) # generate MVAR time courses
        norm_X = np.linalg.norm(X_act, axis = 1) # compute norm 
        ratio = norm_X[-1]/norm_X[0] # retain pairs of time courses whose intensities are close (int_max/int_min < ratio_max)
    
    # normalize the time series so that different pairs have similar intensity
    # (this was important in the paper simulation where more than one pair of time courses were simulated)
    norm_const = np.mean(np.std(X_act, axis = 1))
    X_act = X_act/norm_const 
    
    # compute power spectrum in the freq range of interest
    f, Pwe = signal.welch(X_act, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1)
    P_tot = Pwe[0,:]+Pwe[1,:]
    f_in = np.intersect1d(np.where(f>=fmin)[0],np.where(f<=fmax)[0])
    
    # retain only time courses with sufficiently high power in the frequency range of interest
    if np.sum(P_tot[f_in])/len(f_in)>1.2*np.sum(P_tot)/len(f):
        b,a = signal.butter(3, np.array([8,12]), btype='bandpass', analog=False, output='ba', fs=fs)
        X_act = signal.filtfilt(b, a, X_act, axis=- 1, padtype='odd', padlen=None, method='pad', irlen=None)
        #X_act = mne.filter.filter_data(X_act, 100, 8, 12, picks=None, filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1, method='fir', iir_params=None, copy=True, phase='zero', fir_window='hann', fir_design='firwin', pad='reflect_limited', verbose=False)
        seed_tc[:,:] = X_act
        power_condition = 1
            


# ## Step 2: generate patch activity
# Generate the time courses associated with the sources within the patches, so that they have the desired intracoherence level.
# 

# In[8]:


p1_tcs, p2_tcs = myf.gen_coherent_patches(seed_tc, p1_locs, p2_locs, intra_coh,0, nperseg, nfft, fs, fmin, fmax)
        


# ## Step 3: generate background activtiy (it takes a while)
# To each source outside the patche is assigned a time course following an AR model of order 5. The overall activity of the background sources is then normalized to obtain the desired SNR level
# 

# In[9]:


bg_locs = np.setdiff1d(np.arange(N_dense),np.concatenate((p1_locs,p2_locs))) # locations of background sources
bg_tcs_general = myf.gen_background_tcs(P, len(bg_locs), T) # generate background time courses exploiting an AR model


# define the norm of patches and background activity to define the snr between patches and bg
patches_norm = np.linalg.norm(np.concatenate((p1_tcs,p2_tcs),axis=0), ord='fro')**2
bg_norm_general = np.linalg.norm(bg_tcs_general,ord = 'fro')**2

# scale background time coursed to obtain the desired snr level
bg_tcs = bg_tcs_general*np.sqrt((patches_norm/bg_norm_general)/SNR_background)


# ## Step 4: store the genarate data in a single matrix

# In[10]:


X = np.zeros((N_dense,T))
X[bg_locs,:] = bg_tcs
X[p1_locs,:] = p1_tcs
X[p2_locs,:] = p2_tcs


# # Generate sensor level data

# In[11]:


# generate white gaussian noise
N_tilde = np.random.randn(M,T)

# scale the noise to obtain the desired SNR
Sigma = np.sqrt(np.linalg.norm(G.dot(X), ord='fro')**2/(10**(SNR_sensors/10)*np.linalg.norm(N_tilde, ord='fro')**2))
N = Sigma*N_tilde

# generate sensor level recordings
Y = G.dot(X)+N


# In[12]:


# store data
data = {}
data['X'] = X
data['Y'] = Y
data['sees_loc'] = seed_loc


# # Compute optimal parameters

# In[14]:


# number of parameters to be tested to fintd the optimal one for connectivity estimation
n_lambdas = 15

# initialize the dictionary where to store the parameters
parameters = {'tc':[], 
              'conn': np.zeros((4, n_lambdas)),
              'TPF_conn': np.zeros((n_lambdas,4,20)),
              'FPF_conn': np.zeros((n_lambdas,4,20)),
              'sigma_noise':[]}


# ## Optimal parameter for neural activity estimation

# In[15]:


# define startin point used by the minimize function to find the optimal parameter
input_lamX = np.linalg.norm(N, ord='fro')**2/np.linalg.norm(G.dot(X), ord='fro')**2

# find the optimal parameter
opt_set = optimize.minimize(myf.err_X,input_lamX, args=(X,Y,G,GGt),method='Nelder-Mead')
lamX = opt_set['x'][0].copy()


# ## Optimal parameters for connecttivity estimation

# In[16]:


# define the parameter to be tested to find the optimal one for connectivity estimation, 
# the parameters are defined as multiples of the optimal parameter for neural activity estimation 
lambdas = np.logspace(-5, 1, num = n_lambdas)*lamX

# define the matrix of positives and negatives (positives=1, negtives=0) for connectivity,
# if the (i,j)-th entry of the matrix assumes value 1 (0) indicate that there is (there is not) 
# connection between source i and j
PN_matrix_conn = np.zeros((len(p1_locs),N_dense),dtype = int)
PN_matrix_conn[:,p2_locs] = np.ones((len(p1_locs),len(p2_locs)),dtype = int)
PN_matrix_conn = np.delete(PN_matrix_conn,p1_locs,axis=-1)

# initialize matrices of true positive and false positive fractions
TPF_conn = np.zeros((n_lambdas,4,20)) # dimension: # lambdas x conn_meths x thresholds
FPF_conn = np.zeros((n_lambdas,4,20)) # dimension: # lambdas x conn_meths x thresholds
AUC_conn = np.zeros((n_lambdas,4)) # dimension: # lambdas x conn_meths


# compute the AUC value for each tested parameter and for each connectivity matrix
for i_lam in range(n_lambdas):
    AUC_conn[i_lam,:], TPF_conn[i_lam,:,:], FPF_conn[i_lam,:,:] =    myf.auc(lambdas[i_lam], ['cpsd','imcoh','ciplv','wpli'], G, GGt,
            Y, p1_locs,p2_locs, fmin, fmax, PN_matrix_conn,fs,nperseg)

        


# In[17]:


# store results
parameters['tc'] = lamX.copy()
parameters['conn'][0,:] = AUC_conn[:,0].copy()
parameters['conn'][1,:] = AUC_conn[:,1].copy()
parameters['conn'][2,:] = AUC_conn[:,2].copy()
parameters['conn'][3,:] = AUC_conn[:,3].copy()
parameters['TPF_conn'] = TPF_conn.copy()                
parameters['FPF_conn'] = FPF_conn.copy()


# # Save data in a local folder

# In[18]:



#dir_path = './save_data/single_sim_'+i_sim
#if not op.isdir(dir_path):
#    os.makedirs(dir_path)
                        
#np.save(dir_path+'/parameters.npy', parameters)
#np.save(dir_path+'/data.npy', data)
#np.save(dir_path+'/features.npy', features)


# ## Plots

# In[24]:


# connectivity measures to be plotted

conn_names = ['Cross-power spectrum', 'imCoh', 'ciPLV', 'wPLI']

#i_conn = 0 #cpsd
i_conn = 1 #imCoh
#i_conn = 2 #ciplv
#i_conn = 3 #wpli

# regularization parameter for the selected connectivity meteric
lamC = lambdas[np.argmin(AUC_conn[:,0])]*lamX

# estimated neural activity with the optima parameters lamX and lamC
X_lamX = ((G.T).dot(np.linalg.inv(G.dot(G.T)+lamX*np.eye(M)))).dot(Y)
X_lamC = ((G.T).dot(np.linalg.inv(G.dot(G.T)+lamC*np.eye(M)))).dot(Y)


# ## Plot simulated and estimated neural activity

# In[20]:


# plot simulate brain activity
stc = mne.SourceEstimate(X, vertices=vertno, tmin=0, tstep=1, subject=subject)
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=True, colorbar=True)


# plot reconstructed brain activity (estimateed using lamX)
stc = mne.SourceEstimate(X_lamX, vertices=vertno, tmin=0, tstep=1, subject=subject)
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, 
                   time_viewer=True, colorbar=True)


# plot reconstructed brain activity (estimateed using lamC)
stc = mne.SourceEstimate(X_lamC, vertices=vertno, tmin=0, tstep=1, subject=subject)
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=True, colorbar=True)


# ## Plot simulated and estimated neural activity (absolute value)

# In[42]:


# plot simulated neural activty
stc = mne.SourceEstimate(np.linalg.norm(X,axis=1), vertices=vertno, tmin=0, tstep=1, subject=subject)
title = 'Simulated neural activtiy'
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5, views = 'frontal', 
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=False, colorbar=True)
brain.add_text(0.1, 0.9, title, 'title',font_size=14)


# plot esimated neural activtiy (estimated using lamX)
stc = mne.SourceEstimate(np.linalg.norm(X_lamX,axis=1), vertices=vertno, tmin=0, tstep=1, subject=subject)
title = r'Estimated  neural activtiy (using $\lambda_X$)'
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,views = 'frontal', 
                   hemi=hemi, subjects_dir=subject_dir, 
                   time_viewer=False, colorbar=True)
brain.add_text(0.1, 0.9, title, 'title',font_size=14)


# plot esimated neural activtiy  (estimated using lamC)
stc = mne.SourceEstimate(np.linalg.norm(X_lamC,axis=1), vertices=vertno, tmin=0, tstep=1, subject=subject)
title = r'Estimated neural activtiy (using $\lambda_{'+conn_names[i_conn]+'}$)'
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,views = 'frontal',
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=False, colorbar=True)
brain.add_text(0.1, 0.9, title, 'title',font_size=14)


# ## Plot simulated connectivity and estimated connectivity with both lamX and lamC 

# In[33]:


# Compute true and estimated connectivity

conn_true = np.zeros((len(p1_locs),N_dense))
conn_lamC = np.zeros((len(p1_locs),N_dense))
conn_lamX = np.zeros((len(p1_locs),N_dense))

for i_loc, loc in enumerate(p1_locs):
    
    # true connectivity
    f, Connlam_row = signal.csd(X[loc,:], X, fs=fs, window='hann',
                       nperseg=nperseg, noverlap=nperseg // 2, 
                       nfft=nfft, detrend='constant',
                       return_onesided=True, scaling='density', 
                       axis=-1)
    f_in = np.intersect1d(np.where(f>=fmin)[0],np.where(f<=fmax)[0])
    conn_true[i_loc,:] = np.mean(abs(Connlam_row[:,f_in]), axis=-1)
    
    # estimated connectivity (lamC)
    f, Connlam_row = signal.csd(X_lamC[loc,:], X_lamC, fs=fs, window='hann',
                       nperseg=nperseg, noverlap=nperseg // 2, 
                       nfft=nfft, detrend='constant',
                       return_onesided=True, scaling='density', 
                       axis=-1)
    f_in = np.intersect1d(np.where(f>=fmin)[0],np.where(f<=fmax)[0])
    conn_lamC[i_loc,:] = np.mean(abs(Connlam_row[:,f_in]), axis=-1)
    
    # estimated connectivity (lamX)
    f, Connlam_row = signal.csd(X_lamX[loc,:], X_lamX, fs=fs, window='hann',
                       nperseg=nperseg, noverlap=nperseg // 2, 
                       nfft=nfft, detrend='constant',
                       return_onesided=True, scaling='density', 
                       axis=-1)
    f_in = np.intersect1d(np.where(f>=fmin)[0],np.where(f<=fmax)[0])
    conn_lamX[i_loc,:] = np.mean(abs(Connlam_row[:,f_in]), axis=-1)
    

    
conn_true = np.mean(conn_true,axis=0)
conn_lamX = np.mean(conn_lamX,axis=0)
conn_lamC = np.mean(conn_lamC,axis=0)


# In[37]:


# plot simulate brain connectivity
stc = mne.SourceEstimate(conn_true, vertices=vertno, tmin=0, tstep=1, subject=subject)
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=False, colorbar=True)
title = 'Simulated '+conn_names[i_conn]
brain.add_text(0.1, 0.9, title, 'title',font_size=14)


# plot estimated brain connectivity (estimated using lamX)
stc = mne.SourceEstimate(conn_lamX, vertices=vertno, tmin=0, tstep=1, subject=subject)
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=False, colorbar=True)
title = 'Estimated '+conn_names[i_conn] +r' (with $\lambda_{X}$)'
brain.add_text(0.1, 0.9, title, 'title',font_size=14)


# plot estimated brain connectivity (estimated using lamc)
stc = mne.SourceEstimate(conn_lamC, vertices=vertno, tmin=0, tstep=1, subject=subject)
brain = stc.plot(subject=subject, surface='inflated', smoothing_steps=5,
                   hemi=hemi, subjects_dir=subject_dir, time_viewer=False, colorbar=True)
title = 'Estimated '+conn_names[i_conn] +r' (with $\lambda_{'+conn_names[i_conn]+'}$)'
brain.add_text(0.1, 0.9, title, 'title',font_size=14)


# In[ ]:




