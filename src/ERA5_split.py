from ERA5_preprocessing import load_ERA5, file_path
import os
import numpy as np

precip = load_ERA5(file_path)
I = np.asarray(precip.values)
mask_original = ~np.isnan(I) 

# ----- Mask construction -----
def train_test_split(I, mask_original, seed):
    """Function that constructs the artificial mask
    
    Inputs:
        I (ndaray): Input tensor.
        mask_original (ndarray): Oriinal mask on the data.
        seed (int): For reproducibility.
    Outputs:
        I_true (ndarray): Original tensor.
        I_train (ndarray): Training tensor.
        I_val (ndarray): Validation tensor.
        mask_train (ndarray): Training mask.
        mask_val (ndarray): Validation mask.
"""
    rng = np.random.RandomState(seed)

    observed_entries = np.where(mask_original.ravel())[0]
    observed_entries = rng.permutation(observed_entries)

    # ----- 80/20 split -----
    n_train =int(0.8*len(observed_entries))

    train_entries = observed_entries[:n_train]
    val_entries = observed_entries[n_train:]

    # ----- Mask Construction -----
    mask_train = np.zeros(I.size, dtype=bool)
    mask_val = np.zeros(I.size, dtype=bool)
    mask_train[train_entries] = True
    mask_val[val_entries] = True
    mask_train = mask_train.reshape(I.shape)
    mask_val = mask_val.reshape(I.shape)

    assert not np.any(mask_train & mask_val)
    assert int(mask_train.sum() + mask_val.sum()) == int(mask_original.sum())

    I_true = np.where(mask_original, I, 0.0)

    I_train = I_true.copy()
    I_train[~mask_train] = 0.0

    I_val = I_true.copy()
    I_val[~mask_val] = 0.0

    assert np.all(I_train[mask_val] == 0)
    assert np.all(I_val[mask_train] == 0)

    return I_true, I_train, I_val, mask_train, mask_val

if __name__ == "__main__":
    seed = 456

    I_true, I_train, I_val, mask_train, mask_val = train_test_split(
        I, mask_original, seed
    )

    os.makedirs("data/ERA5_split", exist_ok=True)

    path = "data/ERA5_split/ERA5_training.npz"

    if os.path.exists(path):
        raise SystemError("ERA5 split already exists - delete it explicitly to regenerate.")

    # ----- Metadata -----
    metadata = {
        "seed": np.array(seed),
        "shape": np.array(I.shape),
        "longitude": np.array(precip.longitude.values),
        "latitude": np.array(precip.latitude.values),
        "valid_time": np.array(precip.valid_time.values)
    }

    # ----- Development save -----
    np.savez(path, I_true=I_true, I_train=I_train, I_val=I_val,
             mask_train=mask_train, mask_val=mask_val, **metadata)

    print("ERA5 development and test splits saved.")
