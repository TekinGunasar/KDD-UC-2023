import pickle
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

rank = 10**3

original_data = np.array(pickle.load(open('data_CLA_paradigm.pkl','rb'))[0])

augmented_data = np.array(pickle.load(open('augmented_data_CLA_paradigm.pkl','rb'))[0])
augmented_data = np.reshape(augmented_data,[len(augmented_data),augmented_data.shape[2],augmented_data.shape[1]])

y_original_data = np.array(pickle.load(open('data_CLA_paradigm.pkl','rb'))[1])
y_augmented_data = np.array(pickle.load(open('augmented_data_CLA_paradigm.pkl','rb'))[1])

original_data_right_trials = np.array([original_data[i] for i in range(len(original_data)) if y_original_data[i]==1])
original_data_left_trials = np.array([original_data[i] for i in range(len(original_data)) if y_original_data[i]==2])

augmented_data_right_trials = np.array([augmented_data[i] for i in range(len(augmented_data)) if y_augmented_data[i]==1])
augmented_data_left_trials = np.array([augmented_data[i] for i in range(len(augmented_data)) if y_augmented_data[i]==2])



or_rt = np.reshape(original_data_right_trials,[len(original_data_right_trials),340])
or_lt = np.reshape(original_data_left_trials,[len(original_data_left_trials),340])

ar_rt = np.reshape(augmented_data_right_trials,[len(augmented_data_right_trials),340])
ar_lt = np.reshape(augmented_data_left_trials,[len(augmented_data_left_trials),340])

random_indices_right_original = np.random.randint(size=1000,low=0,high = len(or_rt))
random_indices_left_original = np.random.randint(size=1000,low=0,high=len(or_lt))

random_indices_right_augmented = np.random.randint(size=1000,low=0,high = len(ar_rt))
random_indices_left_augmented = np.random.randint(size=1000,low=0,high = len(ar_lt))

random_right_trials_original = or_rt[random_indices_right_original]
random_right_trials_augmented = ar_rt[random_indices_right_augmented]

cross_corrs = np.zeros((1000, 1000))  # Initialize cross-correlation matrix

for i in range(1000):
    for j in range(1000):
        cur_cross_corr = np.corrcoef(random_right_trials_original[i], random_right_trials_augmented[j])[0][1]
        cross_corrs[i,j] = cur_cross_corr

# Generate color-blind-friendly heat map
cmap = plt.cm.get_cmap("viridis")  # You can choose any other colormap as well
cmap.set_bad(color="lightgray")  # Set the color for invalid (NaN) values

#Plot the heat map
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cross_corrs, cmap=cmap)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Pearson-Correlation', rotation=-90, va="bottom")

# Set axis labels and title
ax.set_xlabel('Original Data')
ax.set_ylabel('Augmented Data')
ax.set_title(f'|r|: {np.mean(np.abs(cross_corrs))}')

# Display the plot
plt.show()















