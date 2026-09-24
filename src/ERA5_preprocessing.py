import xarray as xr
import numpy as np

file_path = "C:/Users/Matthew/OneDrive/Documents/QKTF/data/ERA5_raw.nc"

def load_ERA5(file_path):
    ds = xr.open_dataset(file_path)
    sub_ds = ds.sel(latitude=slice(30, -15),
                    longitude=slice(90,145))
    precip_ds = sub_ds["tp"].transpose("longitude",
                                       "latitude",
                                       "valid_time")

    # ----- Convert precipitation from m to mm -----
    precip_ds = precip_ds*1000
    precip_ds.attrs["units"] = "mm"

    # ----- Convert latitude and longitude into km -----
    lat = precip_ds.latitude.values
    lon = precip_ds.longitude.values
    lat_km = (lat[0] - lat) * 111.32

    reference_lat = np.mean(lat)
    lon_km = (lon - lon[0]) * 111.32 * np.cos(np.radians(reference_lat))

    precip_ds = precip_ds.assign_coords(
        latitude=("latitude", lat_km),
        longitude=("longitude", lon_km)
    )

    precip_ds.attrs["latitude_units"] = "km"
    precip_ds.attrs["longitude_units"] = "km"

    return precip_ds
