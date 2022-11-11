"""
Reproducing paper plots
=======================

Scrivere commenti....
"""

#!/usr/bin/env python
# coding: utf-8

# # 

# In[1]:

import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#get_ipython().run_line_magic('matplotlib', 'notebook')
target  = '..'


# ## Load results

# In[2]:


# it needs to be changed accordingly to were we will store the data
i_sim = 7
features = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/features.npy',allow_pickle='TRUE').item()
tested_par = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/tested_parameters.npy')
seed_tc_loc = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/seed_tc_loc.npy',allow_pickle='TRUE').item()
opt_par = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/opt_par.npy',allow_pickle='TRUE').item()


# ## Extract features

# In[3]:


N_mod = features['N_mod']  # Number of simulated AR models (with connections)
N_act = features['N_act']  # Number of active patches
N_loc = features['N_loc']  # Number of different connected pairs of locations
T = features['T']  # Number of time points
patch_radii = features['patch_radii'] # Patch radius values
coh_levels = features['coh_levels']  # Intracoherence values
bg_noise_levels = features['bg_noise_levels'] # Background SNR values
SNR_val = features['SNR_val'] # Sensor SNR values
N_snr = len(SNR_val) # Number of sensor SNR levels
N_lam = len(tested_par) # Number of tested parameters for connectivity estimation
N_r = len(patch_radii) # Number of radius values
N_c = len(coh_levels) # Number of intracoherence values
N_gamma = len(bg_noise_levels) # Number of background SNR values
N_run = N_mod*N_loc # Number of executed parallel runs

###### this part need to be changed once the codes are done
opt_par = {'tc':np.zeros((N_loc,N_mod,N_r,N_c,N_gamma,N_snr,4)),
           'conn':np.zeros((N_loc,N_mod,N_r,N_c,N_gamma,N_snr,4,N_lam))}

if op.exists('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/opt_par_git.npy'):
    opt_par = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/opt_par_git.npy',allow_pickle='TRUE').item()
else:
    for i_run in range(N_run):
        i_mod = i_run%N_mod
        i_loc = i_run//N_mod
        
        #opt_par_tmp = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/opt_parameters_loc'+str(i_loc)+'_mod'+str(i_mod)+'.npy',allow_pickle='TRUE').item()
        opt_par_tmp = np.load('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/'+str(i_run+1)+'/opt_parameters_loc'+str(i_loc)+'_mod'+str(i_mod)+'_i_run'+str(i_run+1)+'.npy',allow_pickle='TRUE').item()
        opt_par['tc'][i_loc,i_mod,:,:,:,:,:] =  opt_par_tmp['tc']
        opt_par['conn'][i_loc,i_mod,:,:,:,:,:,:] =  opt_par_tmp['conn']
        np.save('/media/mida/Volume/PhD_data/project_canada/sim'+str(i_sim)+'_cluster/opt_par_git.npy', opt_par)
        
        


# ## Bar plots

# In[5]:



# setting the parameters for the plots
params = {'legend.fontsize': 14,
          'lines.linewidth' : 3,
          'figure.figsize': (9,7),
         'axes.labelsize': 16,
         'axes.titlesize': 18,
         'xtick.labelsize':10,
         'ytick.labelsize':10}
width = 0.8*tested_par # width of the bars

i_snr = [0,1,2,3] # sensor SNR to be considered when averaging  
axis = (0,1,2,3,4,5) # axis along with compute the average
fig, ax = plt.subplots(2,2)


mean_cpsd = -opt_par['conn'][:,:,:,:,:,i_snr,0,:].mean(axis=axis)
std_cpsd = -opt_par['conn'][:,:,:,:,:,i_snr,0,:].std(axis=axis)
ax[0,0].bar(tested_par,mean_cpsd, width=width,yerr=std_cpsd) 
ax[0,0].set_ylim([0.4,1])
ax[0,0].set_xscale('log')
ax[0,0].set_title('S')
ax[0,0].set_xlabel('$\lambda/\lambda_x$')
ax[0,0].set_ylabel('mean AUC')
ax[0,0].axvline(1,c='r')
ax[0,0].text(0.29,0.95,'1',rotation=90, c='red')
ax[0,0].grid()


mean_imcoh = -opt_par['conn'][:,:,:,:,:,i_snr,1,:].mean(axis=axis)
std_imcoh = -opt_par['conn'][:,:,:,:,:,i_snr,1,:].std(axis=axis)
ax[0,1].bar(tested_par,mean_imcoh, width=width,yerr=std_imcoh)
ax[0,1].set_ylim([0.4,1])
ax[0,1].set_xscale('log')
ax[0,1].set_title('imCOH')
ax[0,1].set_xlabel('$\lambda/\lambda_x$')
ax[0,1].set_ylabel('mean AUC')
ax[0,1].axvline(1,c='r')
ax[0,1].text(0.29,0.95,'1',rotation=90, c='red')
ax[0,1].grid()
                         

mean_ciplv = -opt_par['conn'][:,:,:,:,:,i_snr,2,:].mean(axis=axis)
std_ciplv = -opt_par['conn'][:,:,:,:,:,i_snr,2,:].std(axis=axis)
ax[1,0].bar(tested_par,mean_ciplv, width=width,yerr=std_ciplv)
ax[1,0].set_ylim([0.4,1])
ax[1,0].set_xscale('log')
ax[1,0].set_title('ciPLV')
ax[1,0].set_xlabel('$\lambda/\lambda_x$')
ax[1,0].set_ylabel('mean AUC')
ax[1,0].axvline(1,c='r')
ax[1,0].text(0.29,0.95,'1',rotation=90, c='red')
ax[1,0].grid()

                            
mean_wpli = -opt_par['conn'][:,:,:,:,:,i_snr,3,:].mean(axis=axis)
std_wpli = -opt_par['conn'][:,:,:,:,:,i_snr,3,:].std(axis=axis)
ax[1,1].bar(tested_par,mean_wpli, width=width,yerr=std_wpli)
ax[1,1].set_ylim([0.4,1])
ax[1,1].set_xscale('log')
ax[1,1].set_title('wPLI')
ax[1,1].set_xlabel('$\lambda/\lambda_x$')
ax[1,1].set_ylabel('mean AUC')
ax[1,1].axvline(1,c='r')
ax[1,1].text(0.29,0.95,'1',rotation=90, c='red')
ax[1,1].grid()
pylab.rcParams.update(params)
fig.tight_layout()



# ## Frequency plots

# In[6]:


lambda_psds = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,:,0,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,:,0]
lambda_imcoh = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,:,1,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,:,0]
lambda_ciplv = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,:,2,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,:,0]
lambda_wpli = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,:,3,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,:,0]

fig, ax = plt.subplots(1,3, tight_layout=True, figsize=(10,4))
shrink = 0.5
origin='lower'
cmap='jet'
hist, yedges, xedges = np.histogram2d(np.log10(np.reshape(lambda_psds,(-1,))) ,np.log10(np.reshape(lambda_wpli,(-1,))), bins=[np.arange(-3,5.5,0.4),np.arange(-3,5.5,0.4)] )
im1 = ax[0].imshow(hist,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap=cmap, origin=origin)
fig.colorbar(im1, ax=ax[0], location='right', shrink=shrink)
ax[0].set_xlabel(r'log$_{10}(\lambda_{wPLI})$')
ax[0].set_ylabel(r'log$_{10}(\lambda_{S})$')
ax[0].set_xticks([-2,0,2,4])
ax[0].set_yticks([-2,0,2,4])


hist, yedges, xedges = np.histogram2d(np.log10(np.reshape(lambda_imcoh,(-1,))) , np.log10(np.reshape(lambda_wpli,(-1,))),bins=[np.arange(-3,5.5,0.4),np.arange(-3,5.5,0.4)])
im2 = ax[1].imshow(hist,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap=cmap, origin=origin)
fig.colorbar(im2, ax=ax[1], location='right', shrink=shrink)
ax[1].set_xlabel(r'log$_{10}(\lambda_{wPLI})$')
ax[1].set_ylabel(r'log$_{10}(\lambda_{imCOH})$')
ax[1].set_xticks([-2,0,2,4])
ax[1].set_yticks([-2,0,2,4])
          
hist, yedges, xedges = np.histogram2d(np.log10(np.reshape(lambda_ciplv,(-1,))),np.log10(np.reshape(lambda_wpli,(-1,))), bins=[np.arange(-3,5.5,0.4),np.arange(-3,5.5,0.4)])
im3 = ax[2].imshow(hist, cmap=cmap, extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], origin=origin)
fig.colorbar(im3, ax=ax[2], location='right', shrink=shrink)
ax[2].set_xlabel(r'log$_{10}(\lambda_{wPLI})$')
ax[2].set_ylabel(r'log$_{10}(\lambda_{ciPLV})$')
ax[2].set_xticks([-2,0,2,4])
ax[2].set_yticks([-2,0,2,4])

fig.colorbar(im3, ax=ax[2], location='right', shrink=shrink)


# In[7]:



fig, ax = plt.subplots(4,3, tight_layout=True, figsize=(10,13))

shrink = 0.58
origin='lower'
cmap='jet'

for i_snr in range(N_snr):
    lambda_psds = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,i_snr,0,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,i_snr,0]
    lambda_imcoh = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,i_snr,1,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,i_snr,0]
    lambda_ciplv = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,i_snr,2,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,i_snr,0]
    lambda_wpli = tested_par[np.argmax(-opt_par['conn'][:,:,:,:,:,i_snr,3,:],axis=-1)]*opt_par['tc'][:,:,:,:,:,i_snr,0]

    hist, yedges, xedges = np.histogram2d(np.log10(np.reshape(lambda_psds,(-1,))) ,np.log10(np.reshape(lambda_wpli,(-1,))), 
                                          bins=[np.arange(-3,5.5,0.4),np.arange(-3,5.5,0.4)] )
    im1 = ax[i_snr,0].imshow(hist,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap=cmap, origin=origin)
    fig.colorbar(im1, ax=ax[i_snr,0], location='right', shrink=shrink)
    ax[i_snr,0].set_xlabel(r'log$_{10}(\lambda_{wPLI})$')
    ax[i_snr,0].set_ylabel(r'log$_{10}(\lambda_{S})$')

    hist, yedges, xedges = np.histogram2d(np.log10(np.reshape(lambda_imcoh,(-1,))) ,np.log10(np.reshape(lambda_wpli,(-1,))),
                                          bins=[np.arange(-3,5.5,0.4),np.arange(-3,5.5,0.4)] )
    im2 = ax[i_snr,1].imshow(hist,extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], cmap=cmap, origin=origin)
    fig.colorbar(im2, ax=ax[i_snr,1], location='right', shrink=shrink)
    ax[i_snr,1].set_xlabel(r'log$_{10}(\lambda_{wPLI})$')
    ax[i_snr,1].set_ylabel(r'log$_{10}(\lambda_{imCOH})$')

    hist, yedges, xedges = np.histogram2d(np.log10(np.reshape(lambda_ciplv,(-1,))) ,np.log10(np.reshape(lambda_wpli,(-1,))), 
                                          bins=[np.arange(-3,5.5,0.4),np.arange(-3,5.5,0.4)] )
    im3 = ax[i_snr,2].imshow(hist, cmap=cmap, extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], origin=origin)
    fig.colorbar(im3, ax=ax[i_snr,2], location='right', shrink=shrink)
    ax[i_snr,2].set_xlabel(r'log$_{10}(\lambda_{wPLI})$')
    ax[i_snr,2].set_ylabel(r'log$_{10}(\lambda_{ciPLV})$')

    
    
    ax[i_snr,1].set_title(r'SNR='+str(np.round(SNR_val[i_snr],decimals=1))+'\n')
    


# In[ ]:




