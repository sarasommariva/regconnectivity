#!/usr/bin/env python
# coding: utf-8

"""
Part 1: Download data
============================================
Download data to run the codes used to simulate MEG data and compute the
parameters, as it was done in the paper
"""


import mne
import os.path as op
import os
import pooch

subject_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
subject = 'sample'

###############################################################################
# Download data (if not already downloaded or generated with generate_fwd.py)

data_path = op.join('.', 'data')
if not op.exists(data_path):
    os.mkdir(data_path)

fname = 'cortico_dist_oct6.npy'
if not op.exists(op.join(data_path, fname)):
    url = 'https://osf.io/download/37kaz/?direct%26mode=render'

    pooch.retrieve(url=url, known_hash=None, path=data_path, fname=fname)


fname = 'oct6_fwd.fif'
if not op.exists(op.join(data_path, fname)):
    url = 'https://osf.io/download/7dfvm/?direct%26mode=render'

    pooch.retrieve(url=url, known_hash=None, path=data_path, fname=fname)
