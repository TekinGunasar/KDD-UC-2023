from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score,normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis

import numpy as np
import pickle

from tensorflow.keras.models import load_model

encoder = load_model('DEC_results_dim_24_epochs_equal_8.h5')

original_data = pickle.load(open('data_CLA_paradigm.pkl','rb'))
augmented_data = pickle.load(open('augmented_data_CLA_paradigm.pkl','rb'))[0]

validation_data = pickle.load(open('test_data_CLA.pkl','rb'))[0]
validation_data = np.array(validation_data)

y_val = pickle.load(open('test_data_CLA.pkl','rb'))[1]

X_original = np.array(original_data[0])
X_augmented = np.reshape(augmented_data,[len(augmented_data),2,170])

y_original = np.array(original_data[1])
y_augmented = np.array(pickle.load(open('augmented_data_CLA_paradigm.pkl','rb'))[1])
y_train = np.concatenate((y_original,y_augmented))

X = np.concatenate((X_original,X_augmented))

X_flattened = np.reshape(X,[len(X),340])
validation_data_flattened = np.reshape(validation_data,[len(validation_data),340])

X_train = MinMaxScaler().fit_transform(X_flattened)
X_val_normalized = MinMaxScaler().fit_transform(validation_data_flattened)

X_train = np.reshape(X_train,[len(X_train),2,170])
X_val_normalized = np.reshape(X_val_normalized,[len(X_val_normalized),2,170])

X_train_reduced = encoder(X_train)[:8000]
X_val_reduced = encoder(X_val_normalized)

print(X_train_reduced.shape,X_val_reduced.shape)

X_train_reduced = TSNE(perplexity=30,n_iter=500).fit_transform(X_train_reduced)
X_val_reduced = TSNE(perplexity=30,n_iter=500).fit_transform(X_val_reduced)

def get_acc(model):

    fit_model = model.fit(X_train_reduced,y_train[:8000])
    y_pred = fit_model.predict(X_val_reduced)
    acc = accuracy_score(y_val,y_pred)
    print(acc)

model = SVC(kernel='linear')
get_acc(model)
