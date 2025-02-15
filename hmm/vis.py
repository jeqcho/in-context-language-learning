#%%
"""
Cheap sketchpad
"""
import matplotlib.pyplot as plt
import numpy as np

# Extract the data
data = {
    (100, 100, 10): 4.568918704986572,
    (200, 300, 10): 4.573798656463623,
    (200, 300, 20): 4.092533588409424,
    (200, 300, 40): 4.022435665130615,
    (200, 600, 10): 4.573096752166748,
    (200, 600, 40): 4.021239757537842,
    (100, 100, 2): 4.5793657302856445,
    (300, 300, 10): 4.573070526123047
}

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# For States vs Loss - group by (seq_length, epochs)
groups_states = {}
for (state, seq_len, epoch), loss in data.items():
    key = (seq_len, epoch)
    if key not in groups_states:
        groups_states[key] = []
    groups_states[key].append((state, loss))

# For Sequence Length vs Loss - group by (states, epochs)
groups_seqlen = {}
for (state, seq_len, epoch), loss in data.items():
    key = (state, epoch)
    if key not in groups_seqlen:
        groups_seqlen[key] = []
    groups_seqlen[key].append((seq_len, loss))

# For Epochs vs Loss - group by (states, seq_length)
groups_epochs = {}
for (state, seq_len, epoch), loss in data.items():
    key = (state, seq_len)
    if key not in groups_epochs:
        groups_epochs[key] = []
    groups_epochs[key].append((epoch, loss))

# Plot States vs Loss
for i, ((seq_len, epoch), points) in enumerate(groups_states.items()):
    x, y = zip(*points)
    ax1.scatter(x, y, label=f'seq_len={seq_len}, epoch={epoch}')
ax1.set_xlabel('Number of States')
ax1.set_ylabel('Loss')
ax1.set_title('States vs Loss')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot Sequence Length vs Loss
for i, ((state, epoch), points) in enumerate(groups_seqlen.items()):
    x, y = zip(*points)
    ax2.scatter(x, y, label=f'states={state}, epoch={epoch}')
ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('Loss')
ax2.set_title('Sequence Length vs Loss')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot Epochs vs Loss
for i, ((state, seq_len), points) in enumerate(groups_epochs.items()):
    x, y = zip(*points)
    ax3.scatter(x, y, label=f'states={state}, seq_len={seq_len}')
ax3.set_xlabel('Number of Epochs')
ax3.set_ylabel('Loss')
ax3.set_title('Epochs vs Loss')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(w_pad=5)
plt.show()
# %%
