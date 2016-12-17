"""
Utility function for the OSG demo
"""

import numpy as np
import os
from misshapen import nonshape

        
def save_Ps_and_Ts(data_filename, Fs = 1000, f_range = (6,12)):
    """
    Saves the indices corresponding to oscillatory peaks and trough into a new numpy file
    """
    
    # Load data
    x = np.load(data_filename)
    
    # Calculate peaks and troughs
    Ps, Ts = nonshape.findpt(x, f_range, Fs = Fs)

    # Save peaks and troughs
    save_dict =  {'Ps':Ps, 'Ts':Ts}
    for key in save_dict.keys():
        filename_save = './out/'+key+'_'+os.path.basename(data_filename)
        np.save(filename_save, save_dict[key])
            
