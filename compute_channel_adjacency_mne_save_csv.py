# %%
import mne
import pandas as pd
import numpy as np
from mne.channels import find_ch_adjacency

# # fif epoch filename
# epoch_fif = 'sub-013_ses-mecha_task-NT_epochs.fif' # example path
# # load epoch
# epochs = mne.read_epochs(epoch_fif, preload=True)
# # load adjacency
# adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type="eeg")

# # Magda's data
raw_fif = '/Users/mg/Desktop/MOVE/resting_state/pre-processed_data/subj001/pre-proc_001_rest1.fif'
raw = mne.io.read_raw(raw_fif, preload=True)
adjacency, ch_names = find_ch_adjacency(raw.info, ch_type="eeg")
mne.viz.plot_ch_adjacency(raw.info, adjacency, ch_names)


# %% save csv files of adjacency

montage = raw.get_montage()
ch_pos = montage.get_positions()['ch_pos']  # dict name -> (x,y,z)

# create an Info that contains those channels so find_layout can compute topomap coords
ch_names = list(ch_pos.keys())
sfreq = 1.0
info_tmp = mne.create_info(ch_names, sfreq, ch_types='eeg')
info_tmp.set_montage(montage)

# find_layout returns a Layout object with .pos (2D meters) and .names
layout = mne.channels.find_layout(info_tmp)  # uses topomap coords

pos_map = {name: tuple(pos) for name, pos in zip(layout.names, layout.pos)}

positions = []
for name in ch_names:  # your desired ordering
    if name in pos_map:
        val = pos_map[name]
        x, y = float(val[0]), float(val[1])

        positions.append({'ch_name': name, 'x': float(x), 'y': float(y)})
    else:
        print(f"Warning: No position for {name} — using (0,0)")
        positions.append({'ch_name': name, 'x': 0.0, 'y': 0.0})

pd.DataFrame(positions).to_csv('channel_positions_Magda.csv', index=False)


# %%
adj_df = pd.DataFrame(adjacency.toarray(), index=ch_names, columns=ch_names)
adj_df.to_csv('adjacency.csv')

print("Exported: channel_positions.csv, adjacency.csv")
print(f"EEG channels: {len(ch_names)}")