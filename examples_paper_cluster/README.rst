.. _paper:

Run on cluster
================

The following scripts were run on clusters to obtain the results presented in our paper [1]_.

.. warning::
    These codes are not meant to run in a laptop.

However if you wish to downoload them and run, we suggest to follows the instruction below.
Download all codes in this page in the same folder. Then, add to the folder the reqirments.txt file at this `link <https://github.com/theMIDAgroup/regconnectivity/blob/main/requirements.txt>`_. 
Finally, run the following commands in a terminal, where N_mod is the number of pairs of time courses used to simulate patches' activity, N_loc is the number of source locations and i_job is a number in the range 1-N_mod*N_loc which indicates the combination of source model and source position. Ideally all jobs are to be run in parallel on a cluster.

.. warning::
    These core were run with MNE-python 0.22, more recent versions might be incompatible.

.. code::

	python3 -m venv regconn_env
	source regconn_env/bin/activate
	# In Windows replace the previous command with
	# regconn_env\Scripts\activate 
	pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	python3 00_download_data.py
	python3 01_generate_seed_tc.py N_mod N_loc
	python3 02_sim_realistic_dense.py N_mod N_loc i_job
	

.. [1] ref to the paper



