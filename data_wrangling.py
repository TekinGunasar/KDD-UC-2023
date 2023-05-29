from scipy.io import loadmat
import numpy as np
import os
from tsaug import AddNoise,TimeWarp

import pickle

def load_mat(path):
    return loadmat(path)

def get_markers(path):
    #Reshaping markers to be in a more convenient form
    markers = load_mat(path)['o']['marker'][0][0]
    return np.reshape(markers,[markers.size,])

def get_sampling_rate(path):
    return load_mat(path)['o']['sampFreq'][0][0][0][0]

def get_raw_eeg_data(path):
    return load_mat(path)['o']['data'][0][0]

def get_input_lead_organization(path):
    return load_mat(path)['o'][0][0][7]
    #C3 located at index 4
    #C4 located at index 5

def get_signal_onsets(path):
    markers = get_markers(path)
    sampling_rate = get_sampling_rate(path)

    signal_onset_indices = []
    signal_onset_type = []

    lastFound = 0
    for i in range(len(markers)):

        if markers[i] == 0 or markers[i] == 3:
            lastFound += 1
            continue

        elif markers[i] != 0 and lastFound > sampling_rate:
            lastFound = 0
            signal_onset_indices.append(i)
            signal_onset_type.append(markers[i])

    return signal_onset_indices,signal_onset_type

def get_C3_C4(path):
    raw_eeg = get_raw_eeg_data(path)
    C3 = raw_eeg[:,4]
    C4 = raw_eeg[:,5]
    C3_C4_eeg = np.array([C3,C4])

    return C3_C4_eeg

def get_mi_trials(path,trial_length=0.85):
    mi_trials = []

    C3_C4_eeg = get_C3_C4(path)

    signal_onsets = get_signal_onsets(path)
    signal_onset_indices = signal_onsets[0]
    labels = signal_onsets[1]

    sampling_rate = get_sampling_rate(path)

    for index in signal_onset_indices:
        trial_start = index
        trial_end = trial_start + int(sampling_rate * trial_length)

        cur_trial_eeg = C3_C4_eeg[:,trial_start:trial_end]

        mi_trials.append(cur_trial_eeg)

    return mi_trials,labels

def create_dataset(data_dir,save_to,trial_length=0.85):
    dataset_dir = os.listdir(data_dir)
    X = []
    y = []

    for file in dataset_dir:

        try:

            cur_data_path = os.path.join(data_dir, file)
            print(f'Reading {cur_data_path}')
            trials_and_labels = get_mi_trials(cur_data_path, trial_length)

            cur_mi_trials = trials_and_labels[0]
            cur_labels = trials_and_labels[1]

            X += cur_mi_trials
            y += cur_labels

        except:
            print(f'Error Reading {cur_data_path}')

    with open(save_to,'wb') as pickle_file:
        pickle.dump((X,y),pickle_file)


def augment_dataset(X,y=None):

    augmentor = (TimeWarp(n_speed_change=3,max_speed_ratio=1.1,repeats=4)
            + AddNoise(loc=0,scale=[0.4,0.5,0.6],prob=0.8,repeats=2))

    return augmentor.augment(X,y)


#if __name__ == '__main__':
#    data_dir = 'CLA Datasets'
#    save_to = 'data_CLA_paradigm.pkl'

#    create_dataset(data_dir,save_to)














