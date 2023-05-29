from tensorflow.keras.models import load_model
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,LocallyLinearEmbedding,SpectralEmbedding
from sklearn.cluster import KMeans,SpectralClustering
from tensorflow.keras import Sequential
from sklearn.neighbors import DistanceMetric
import tensorflow as tf

from sklearn.metrics import normalized_mutual_info_score

from scipy.spatial import distance

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


encoder = load_model('DEC_results_dim_24_epochs_equal_7.h5')

validation_data = pickle.load(open('test_data_CLA.pkl','rb'))
X_val = np.array(validation_data[0])
y_val = np.array(validation_data[1])

X = np.reshape(X_val,[len(X_val),340])
X = MinMaxScaler().fit_transform(X)
X = np.reshape(X,[len(X),2,170])

X = np.expand_dims(X,axis=3)
embeddings = tf.cast(encoder(X)[:2000],tf.float64)

cluster_centers = pickle.load(open('cluster_centers_results_24_dim_epochs_equal_7.pkl','rb'))
reduced_embeddings = TSNE(perplexity=30,n_iter=5000,n_jobs=8).fit_transform(np.concatenate((embeddings,cluster_centers)))

r_mu_1,r_mu_2 = reduced_embeddings[0],reduced_embeddings[1]
reduced_embeddings = reduced_embeddings[:-2]

mu_1,mu_2 = tf.cast(cluster_centers[0],tf.float64),tf.cast(cluster_centers[1],tf.float64)

#r_mu_1,r_mu_2 = TSNE(perplexity=30,n_iter=250,n_jobs=8).fit_transform(cluster_centers)
print(r_mu_1,r_mu_2)

dist_centroids = tf.linalg.norm(r_mu_1 - r_mu_2)
print(f'Distance between centroids {dist_centroids}')

class_one = []
class_two = []

num_correct_predictions = 0

y_pred = []

for i in range(len(embeddings)):
    embedding = reduced_embeddings[i]
    dist_one = distance.euclidean(r_mu_1,embedding)
    dist_two = distance.euclidean(r_mu_2,embedding)

    if dist_one < dist_two:

        class_one.append(embedding)
        if y_val[i] == 1:
            num_correct_predictions += 1

    else:
        class_two.append(embedding)
        if y_val[i] == 2:
            num_correct_predictions += 1

acc = num_correct_predictions / 2000

print(acc)
print(len(class_one),len(class_two))

plt.scatter(np.array(class_one)[:,0],np.array(class_one)[:,1],alpha=0.4)
plt.scatter(np.array(class_two)[:,0],np.array(class_two)[:,1],alpha=0.4)

plt.scatter(r_mu_1[0],r_mu_1[1],s=240)
plt.scatter(r_mu_2[0],r_mu_2[1],s=240)

plt.legend(['Left Hand','Right Hand','Cluster Center(RH)','Cluster Center(LH)'],fontsize=15)

plt.show()

