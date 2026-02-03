# %%
import os 
import mne 
import numpy as np
from mne.channels import find_ch_adjacency
from scipy.io import savemat, loadmat

# fif epoch filename
epoch_fif = 'sub-013_ses-mecha_task-NT_epochs.fif' # example path
        
# load epoch 
epochs = mne.read_epochs(epoch_fif, preload=True)

# load adjacency
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type="eeg")

# get the data from the adjacency matrix into arrays
adj_array = adjacency.nonzero()
label_idx = adj_array[0]
neighbour_idx = adj_array[1]

neighbours = []
labels = []

ch_names = np.array(ch_names)

# put the neighbour for each channel in separate lists
for i, ch in enumerate(ch_names):

    print(i, ch)
    current_label_idx = label_idx==i
    current_neighb_idx = neighbour_idx[current_label_idx]
    neighb_labels = ch_names[current_neighb_idx]

    # save in dict
    neighbours.append(neighb_labels)


# save them into .mat file
dtype = [('label', 'O'), ('neighlabel', 'O')]
fields = np.empty((1, len(ch_names)), dtype=dtype)
fields['label'] = ch_names
fields['neighlabel'] = neighbours

savemat('channel_adjacency.mat', {'neighbours': fields})
