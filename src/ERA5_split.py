from ERA5_preprocessing import load_ERA5, file_path
import os
import numpy as np

precip = load_ERA5(file_path)
I = np.array(precip.values)
mask_original = ~np.isnan(I) 

# ----- Mask construction -----
def train_test_split(I, mask_original, seed):
    """Function that constructs the artificial mask
    
    Inputs:
        I (ndaray): Input tensor.
        mask_original (ndarray): Oriinal mask on the data.
        seed (int): For reproducibility.
    Outputs:
        I_true (ndarray): 
        I_train (ndarray):
        I_val (ndarray):
        mask_train (ndarray):
        mask_val (ndarray):
"""

