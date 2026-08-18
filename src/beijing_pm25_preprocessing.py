import pandas as pd
import os
import numpy as np

folder = "Data/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228"

def load_beijing_pm25(folder):
    files = [os.path.join(folder, f) # joins the path 'folder/f'
            for f in os.listdir(folder)] # iterates through the names of entries in directory 'folder'
    dfs = [pd.read_csv(f) for f in files] # creates a dataframe of all the files
    df = pd.concat(dfs, ignore_index=True) # concatenates into a single dataframe

    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]]) # constructs a new column called datetime
    df = df[(df["datetime"] < "2014-03-01") & (df["datetime"] >= "2013-03-01")] # filters the dataframe to have one exact year
    pm25 = df[["day", "hour", "PM2.5", "station", "datetime"]].copy()

    # ----- Adding longitude and latitude coordinates -----
    station_coords = {
        "Aotizhongxin": (116.397, 39.982),
        "Changping": (116.23, 40.217),
        "Dingling": (116.22, 40.292),
        "Dongsi": (116.417, 39.929),
        "Guanyuan": (116.339, 39.929),
        "Gucheng": (116.184, 39.914),
        "Huairou": (116.628, 40.328),
        "Nongzhanguan": (116.461, 39.937),
        "Shunyi": (116.655, 40.127),
        "Tiantan": (116.407, 39.886),
        "Wanliu": (116.287, 39.987),
        "Wanshouxigong": (116.352, 39.878)
    }

    pm25["longitude"] = pm25["station"].map(lambda station: station_coords[station][0])
    pm25["latitude"] = pm25["station"].map(lambda station: station_coords[station][1])

    pm25 = pm25.sort_values(["station", "datetime"]).reset_index(drop=True) # sorts dataframe

    # ----- Create a sequential ordering of days -----
    pm25["date"] = pm25["datetime"].dt.normalize()
    dates = sorted(pm25["date"].unique())
    date_index = {date: i for i, date in enumerate(dates)}
    pm25["day"] = pm25["date"].map(date_index)

    return pm25, station_coords

# ----- Haversine distance for covariance functions -----
def haversine(station_one, station_two, station_coords):
    """
    Calculates the Haversine distance between two points (measured via longitude and latitude).
    
    Inputs:
        station_one (string): name of first station
        station_two (string): name of second station
        coords (tuple): dictionary containing (longitude, latitude)
    Outputs:
        float: haversine distance between the two coordinates"""
    radius = 6367.45 # mean radius of Earth in km

    long_one, lat_one = station_coords[station_one]
    long_two, lat_two = station_coords[station_two]

    long_onerad = np.deg2rad(long_one)
    lat_onerad = np.deg2rad(lat_one)
    long_tworad = np.deg2rad(long_two)
    lat_tworad = np.deg2rad(lat_two)

    dist_long = long_tworad - long_onerad
    dist_lat = lat_tworad - lat_onerad

    a = np.sin(dist_lat/2)**2 + (np.cos(lat_onerad) * np.cos(lat_tworad)  * np.sin(dist_long/2)**2)
    d = (2 * radius) * np.arcsin(np.sqrt(a))

    return d

# ----- Computing the distance matrix -----
def station_dist(pm25, station_coords):
    stations = sorted(pm25["station"].unique())
    d_station = np.zeros((len(stations), len(stations)))

    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            distance = haversine(stations[i], stations[j], station_coords)
            d_station[i, j] = distance
            d_station[j, i] = distance

    return d_station

