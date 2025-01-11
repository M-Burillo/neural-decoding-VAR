#Copyright 2024 Marc Burillo

#This code is not authorized for use, copying, modification, or distribution without explicit permission from the author.

#This code extracts the high-frequency component (200-500Hz) with a Butterworth bandpass filter. Additionally prepares the CV method used throughout the rest of the project. 



from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import hdf5storage
import pandas as pd
import matplotlib.pyplot as plt


# %% Definitions
@dataclass
class lfpData:
    raw: np.ndarray
    epoched: np.ndarray
    param_dict: Dict[str, Any]

    def __post_init__(self):
        self.ne = self.epoched.shape[0]  # number of electrodes
        self.ntp = self.epoched.shape[1]  # number of time points (epoched)
        self.nt = self.epoched.shape[2]  # number of trials
        self.nc = self.epoched.shape[3]  # number of classes
        self.ntp_raw = self.raw.shape[1]  # number of time points (raw)
def import_mat(fn):
    data = hdf5storage.loadmat(fn)
    raw = data["data"]["raw"][0][0]
    epoched = data["data"]["epoched"][0][0]
    event_times = data["data"]["event_times"][0][0]
    event_names = data["data"]["event_names"][0][0]
    epoch_inds = data["data"]["epoch_inds"][0][0]
    param_dict = {}
    param_dict["hand"] = data["data"]["hand"][0][0][0]
    param_dict["spikes"] = data["data"]["spikes"][0][0][0][0][0]
    param_dict["angle"] = data["data"]["angle"][0][0][0]
    param_dict["event_times"] = event_times
    param_dict["event_names"] = event_names
    param_dict["epoch_inds"] = epoch_inds
    return lfpData(raw, epoched, param_dict)



def import_coordinates():
    brain_regions = pd.read_csv(mat_dir + "/regions.csv")
    brain_regions_dict = {
        row["region"]: np.arange(row["start_electrode"] - 1, row["end_electrode"])
        for (idx, row) in brain_regions.iterrows()
    }
    # import electrode coordinates
    electrode_coordinates = pd.read_csv(
        mat_dir + "/coordinates.csv", sep="\t", header=None
    )
    electrode_coordinates.columns = ["e", "x", "y", "z"]

    # combine brain regions and coordinates
    region_col = []
    x_col = []
    y_col = []
    for i, row in brain_regions.iterrows():
        region_col += [row["region"]] * (
            row["end_electrode"] - row["start_electrode"] + 1
        )
        x_col += list(
            electrode_coordinates.loc[
                row["start_electrode"] - 1 : row["end_electrode"] - 1, "x"
            ]
        )
        y_col += list(
            electrode_coordinates.loc[
                row["start_electrode"] - 1 : row["end_electrode"] - 1, "y"
            ]
        )
    electrodes = pd.DataFrame({"region": region_col, "x": x_col, "y": y_col})
    electrodes.insert(0, "ei", electrodes.index)
    return electrodes
    
# %% User-defined parameters
data_dir = (
    "animal_1/"
)

mat_dir = data_dir
mat_fn = '1_data_left_0.mat'
save_dir = data_dir + "/python_output/"

# %% Import data
data = import_mat(mat_dir + mat_fn)
electrodes = import_coordinates()
np.shape(data.epoched)
# %%

# %% User-defined parameters
data_dir = (
    "animal_1/"
)


# In the current study, we combine both hand and angles as equivalent conditions
mat_dir = data_dir
mat_fn = ['1_data_left_0.mat', '2_data_left_45.mat', '3_data_left_90.mat', '4_data_left_135.mat', '5_data_right_0.mat',
          '6_data_right_45.mat', '7_data_right_90.mat', '8_data_right_135.mat']
save_dir = data_dir + "/python_output/"
for i in range(8):
    # %% Import data
    data = import_mat(mat_dir + mat_fn[i])
    electrodes = import_coordinates()
    el = data[1][v]
    Nel = np.size(v)
    sf = 2034.5
    lowcut = 200
    highcut = 500
    print(np.shape(el))
    clean = bandpass_filter_epoched(el, sf, lowcut, highcut)
    if i== 0:
        rel = np.transpose(clean, axes=[2,3,1,0])
        print(np.shape(rel))
    else:
        rel = np.append(rel,np.transpose(clean, axes=[2,3,1,0]),axis=0)
# %%
permutation = np.random.permutation(rel.shape[0])

# Shuffle the array along the first axis (samples axis) using the permutation
np.random.shuffle(rel)

np.save('310_filtered.npy',rel[0:310,:,:,:]) #to use a 10-Fold method of equal size folds we will spare 5 trials


def partition_list(lst, n):
    if n <= 0 or len(lst) % n != 0:
        raise ValueError("Error.")

    random.shuffle(lst)

    subset_size = len(lst) // n

    partitions = [lst[i * subset_size:(i + 1) * subset_size] for i in range(n)]

    return partitions


llista = np.arange(310)
num_csets = 10

partitions = [partition_list(llista, num_sets)]
for i in range(1,30):
    partitions=np.vstack((partitions,[partition_list(llista, num_sets)]))

np.save('30_10fold.npy',particions)
