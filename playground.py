import pickle
import numpy as np
from data_wrangling import *

from sklearn.model_selection import train_test_split

dataset = pickle.load(open('data_CLA_paradigm_filtered_8_13.pkl','rb'))
X = dataset[0]

test_split = 0.2
#reshaping to make it work with TSAUG -> (N,T,C)
X = np.reshape(X,[9897,170,2])[:-1]

y = np.array(dataset[1])[:-1]

#want to make sure we have a testing set that is not used at all for augmentation purposes
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_split)

y_aug = []

for i in range(len(X_train)):

    cur_label = y_train[i]
    aug_label = np.ones(X_train[i].shape) * cur_label
    y_aug.append(aug_label)

y_aug = np.array(y_aug)

augmented_dataset = augment_dataset(X_train,y_aug)

X_aug = augmented_dataset[0]
y_new = []

for label in augmented_dataset[1]:
    y_new.append(label[0][0])

with open('augmented_data_CLA_paradigm_filtered_8_13.pkl','wb') as pickle_file:
    pickle.dump((X_aug,y_new),pickle_file)

with open('test_data_CLA_filtered_8_13.pkl','wb') as pickle_file:
    pickle.dump((X_test,y_test),pickle_file)



