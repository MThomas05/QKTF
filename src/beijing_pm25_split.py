from beijing_pm25_preprocessing import load_beijing_pm25, folder, station_dist
import pandas as pd
import os
import numpy as np

pm25, station_coords = load_beijing_pm25(folder)
d_station = station_dist(pm25, station_coords)

# ----- Tensor construction (station x day x hour) -----
stations = sorted(pm25["station"].unique())
coords = np.array([station_coords[station] for station in stations], dtype=np.float64)
hours = list(range(24))
days = sorted(pm25["day"].unique())

index = pd.MultiIndex.from_product([stations, days, hours], names=["stations", "days", "hours"])
pm25_index = (pm25.set_index(["station", "day", "hour"])["PM2.5"].reindex(index))

I_numpy = pm25_index.to_numpy().reshape(len(stations), len(days), len(hours))
I = np.array(I_numpy) # complete input tensor

mask_original = ~np.isnan(I) # corresponds to the geninely observed tensor

# ----- Mask construction -----
def train_test_split(I, mask_original, seed):
    """
    Function that constructs the artifical mask
    
    Inputs:
        I (ndarray): input tensor
        mask_original (ndarray): original mask on the data - masks NaN values
        seed (int): for reproducibility
    Outputs:
        mask (ndarray): artificial mask used for training and testing across all algorithms
        I (ndarray): input tensor"""
    rng = np.random.RandomState(seed)

    observed_entries = np.where(mask_original.ravel())[0]
    observed_entries = rng.permutation(observed_entries)

    # ----- 80/20 split -----
    n_train = int(0.8*len(observed_entries))

    train_entries = observed_entries[:n_train]
    val_entries = observed_entries[n_train:]

    # ---- Mask construction -----
    mask_train = np.zeros(I.size, dtype=bool)
    mask_val = np.zeros(I.size, dtype=bool)
    mask_train[train_entries] = True
    mask_val[val_entries] = True
    mask_train = mask_train.reshape(I.shape)
    mask_val = mask_val.reshape(I.shape)

    assert not np.any(mask_train & mask_val)
    assert int(mask_train.sum() + mask_val.sum()) == int(mask_original.sum())

    # ----- True tensor -----
    I_true = np.where(mask_original, I, 0.0)

    # ----- Development tensor -----
    I_train = I_true.copy()
    I_train[~mask_train] = 0.0

    # ----- Testing tensor -----
    I_val = I_true.copy()
    I_val[~mask_val] = 0.0

    assert np.all(I_train[mask_val] == 0)
    assert np.all(I_val[mask_train] == 0)

    return I_true, I_train, I_val, mask_train, mask_val

if __name__ == "__main__":
    seed = 123

    I_true, I_train, I_val, mask_train, mask_val = train_test_split(
        I, mask_original, seed
    )

    os.makedirs("Data/splits", exist_ok=True)

    path = "Data/splits/beijing_pm25_training.npz"

    if os.path.exists(path):
        raise SystemExit("Beijing split already exists - delete it explicitly to regenerate.")

    # ----- Metadata -----
    metadata = {
        "seed": np.array(seed),
        "shape": np.array(I.shape),
        "stations": np.array(stations),
        "hours": np.array(hours),
        "days": np.array(days),
        "station_coords": coords,
        "coord_names": np.array(["longitude", "latitude"]),
        "d_station": d_station,
        "distance_units": np.array("km")
    }

    # ----- Development save -----
    np.savez(path, I_true=I_true, I_val=I_val, I_train=I_train,
            mask_original=mask_original, mask_val=mask_val, mask_train=mask_train, **metadata)

    print("Beijing development and test splits saved.")



