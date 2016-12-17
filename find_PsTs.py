# -*- coding: utf-8 -*-
"""
Script to identify peaks and troughs in a time series
"""

# Process command line argument to define path to data
import sys
cmdargs = str(sys.argv) # Read command line arguments
userid = 'srcole' # My userid, where the data is located
data_filename = '/stash/user/' + userid + '/lfp_set/' + cmdargs[1] + '.npy' # Define path to data

# Find peaks and troughs
import util
util.save_Ps_and_Ts(data_filename) # Execute function to find peaks and troughs