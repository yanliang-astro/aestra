#!/usr/bin/env python
# coding: utf-8

import io, os, sys, time, random
import numpy as np
import pickle
import  multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import gaussian_filter1d

def find_deepest_lines(wave_obs, raw_spectrum, num_lines=30, min_separation=0.10,return_ind=False):
    spectrum = raw_spectrum/np.quantile(raw_spectrum,0.99)
    depth = 1 - spectrum
    sorted_indices = np.argsort(depth)[::-1]  # Indices of depths sorted from largest to smallest

    # Start with the deepest line
    unique_indices = [sorted_indices[0]]

    # Iterate over the sorted indices and select peaks with the required minimum separation
    for index in sorted_indices[1:]:
        # Check if this index is sufficiently far from all previously selected peaks
        if all(np.abs(wave_obs[index] - wave_obs[prev_index]) > min_separation for prev_index in unique_indices):
            unique_indices.append(index)
        # Stop if we've found the required number of lines
        if len(unique_indices) == num_lines:
            break

    # Get the corresponding wavelengths and depths for the unique peaks
    unique_wavelengths = wave_obs[unique_indices]
    unique_depths = depth[unique_indices]

    if return_ind:return sorted(zip(unique_indices, unique_depths), key=lambda x: x[0])
    # Return as a list of tuples sorted by depth
    return sorted(zip(unique_wavelengths, unique_depths), key=lambda x: x[0])

def mask_deepest_lines(wave_obs, lines_to_mask, mask_width=0.15):
    # Copy the spectrum to avoid modifying the original
    skymask = np.zeros_like(wave_obs,dtype=bool)
    # Loop through the lines to mask
    for line in lines_to_mask:
        # Find the indices within the mask width of the line
        mask_indices = np.where(np.abs(wave_obs - line[0]) <= mask_width)[0]
        # Mask the spectrum by setting the depth to 1 (i.e., no absorption)
        skymask[mask_indices] = True
    return skymask

cloud_monitor_dir = "/scratch/gpfs/yanliang/NEID-PYRHELIO"
files = sorted([item for item in os.listdir(cloud_monitor_dir) if ".tel" in item])
cloud_dict = {}
#with open("pyrohelio.pkl","rb") as f:
#    cloud_dict = pickle.load(f)
#print("cloud_dict:",cloud_dict.keys())


for i,file in enumerate(files):
    file_path = "%s/%s"%(cloud_monitor_dir,file)
    print("file_path:",file_path)
    # Read the data from the text file
    solar_irr = pd.read_csv(file_path, sep=" ", header=None, names=["Time", "Photocell_Output", "Solar_Irradiance"])
    # Convert the time column to datetime
    solar_irr["Time"] = pd.to_datetime(solar_irr["Time"])
    # Convert the datetime to Julian dates
    solar_irr["JulianDate"]=Time(solar_irr["Time"].values).jd
    jd_irr = np.array(solar_irr["JulianDate"])
    irr = np.array(solar_irr["Solar_Irradiance"])
    irr[np.isnan(irr)] = 0

    noon = np.abs((jd_irr%1)-0.3)<0.15

    for jd in np.unique(jd_irr//1.0):
        if jd<2459196:continue
        if "%d"%jd in cloud_dict:
            print("jd",jd,"exists...")
            continue
        mask = noon & ((jd_irr//1.0)==jd)
        if mask.sum()==0:
            print("mask.sum()==0")
            continue
        print("processing jd %d..."%jd)
        clouds = find_deepest_lines(jd_irr[mask], irr[mask], num_lines=100, min_separation=0.005,return_ind=False)
        clouds = [item for item in clouds if item[1]>0.05]
        cloud_dict["%d"%jd] = {"jd":jd_irr[mask],"irr":irr[mask],"clouds":clouds}

    if i%10==0:
        print("Saving...")
        with open("pyrohelio.pkl","wb") as f:
            pickle.dump(cloud_dict,f)

with open("pyrohelio.pkl","wb") as f:
    pickle.dump(cloud_dict,f)