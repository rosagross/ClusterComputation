# %%
import mne
import pandas as pd
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
pos_dict = montage.get_positions()['ch_pos']
positions = []
for name in ch_names:
    if name in pos_dict:
        pos = pos_dict[name]
        positions.append({'ch_name': name, 'x': float(pos[0]), 'y': float(pos[1])})
    else:
        print(f"Warning: No position for {name} — using (0,0)")

pd.DataFrame(positions).to_csv('channel_positions.csv', index=False)


adj_df = pd.DataFrame(adjacency.toarray(), index=ch_names, columns=ch_names)
adj_df.to_csv('adjacency.csv')

print("Exported: channel_positions.csv, adjacency.csv")
print(f"EEG channels: {len(ch_names)}")