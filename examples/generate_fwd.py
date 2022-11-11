"""
Title of the example
===========================
Compute forward and cortical distances
"""
from scipy import sparse
import numpy as np
import os.path as op
import mne
from mne.datasets import sample
from conpy import restrict_forward_to_vertices
data_path = sample.data_path()


#%%
def labels_to_array(labels, src, type_index='python'):
    """Project MNE Label(s) to a given source-space and save as list of arrays
    
    Parameters:
    -----------
    labels : list of Label
        Label(s) to be projected
    src : SourceSpaces
        Source space 
    type_index : string ['python' | 'matlab']
    """
    
    name = [lab.name for lab in labels]
    parcels = list()
    total_vertx = np.ones(src[0]['nuse']+src[1]['nuse'])
    
    for lab in labels:
        if lab.hemi == 'lh':
            tmp_idx = np.nonzero(np.in1d(src[0]['vertno'], lab.vertices))[0]
        elif lab.hemi == 'rh':
            tmp_idx = src[0]['nuse'] + \
                np.nonzero(np.in1d(src[1]['vertno'], lab.vertices))[0]    
        parcels.append(tmp_idx)
        total_vertx[tmp_idx] = 0
        
    outliers = np.nonzero(total_vertx)[0]
    
    if outliers.size > 0:
        parcels.append(outliers) # For consistency with flame parcels
        name.append('Outliers')

    if type_index == 'matlab':
        parcels = map(lambda x:x+1, parcels)
        outliers = outliers + 1
    elif type_index == 'python':
        pass
    else:
        print('Type of indeces not understood')

    converted_labels = {'parcel' : parcels, 'name' : name, 
            'outliers_id' : outliers, 'outliers' : float(outliers.shape[0])}
    
    return converted_labels

#%%
# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'


trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

info = mne.io.read_info(raw_fname)

src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             subjects_dir=subjects_dir)


###############################################################################
# .. _plot_forward_compute_forward_solution:
#
# Compute forward solution

conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                verbose=True)

# Remove anatomical outliers from fwd model

# 4.1. Define anatomical outliers
parc = 'aparc' 
label_lh = mne.read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                                  subjects_dir=subjects_dir)
label_rh = mne.read_labels_from_annot(subject=subject, parc=parc, hemi='rh',
                                  subjects_dir=subjects_dir)
label = label_lh + label_rh

anat_array = labels_to_array(label, fwd['src'], type_index='python')
anat_outliers = anat_array['outliers_id']

nv_lh = fwd['src'][0]['nuse']
out_lh = anat_outliers[np.where(anat_outliers < nv_lh)[0]]
out_rh = anat_outliers[np.where(anat_outliers >= nv_lh)[0]] - nv_lh

sel_vert_lh = np.delete(fwd['src'][0]['vertno'], out_lh)
sel_vert_rh = np.delete(fwd['src'][1]['vertno'], out_rh)

fwd_sel = restrict_forward_to_vertices(fwd, (sel_vert_lh, sel_vert_rh),
                                       check_vertno=True, copy=True, verbose=None)


###############################################################################
fwd_fixed = mne.convert_forward_solution(fwd_sel, surf_ori=True, force_fixed=True,
                                         use_cps=True)

f_name = op.join('.','data','oct6_fwd.fif')
mne.write_forward_solution(f_name,fwd_fixed, overwrite=True)
##############################################################################
# generate and save cortical distances
#%%
src_fixed = fwd_fixed['src']
mne.add_source_space_distances(src_fixed)

lh_idx = fwd_fixed['src'][0]['vertno']
rh_idx = fwd_fixed['src'][1]['vertno']
cortical_dist_lh = src_fixed[0]['dist'][lh_idx,:][:,lh_idx].todense()
cortical_dist_rh = src_fixed[1]['dist'][rh_idx,:][:,rh_idx].todense()
cortical_dist = np.concatenate((np.concatenate((cortical_dist_lh,10*np.ones((len(lh_idx),len(rh_idx)))),axis=1),
                               np.concatenate((10*np.ones((len(rh_idx),len(lh_idx))),cortical_dist_rh),axis=1)),axis=0)
f_name = op.join('.','data','cortico_dist_oct6.npy')
np.save(f_name, cortical_dist)
