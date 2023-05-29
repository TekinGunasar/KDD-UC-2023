from tensorflow.keras.models import load_model
import tensorflow as tf
from DEC_Training_Helper_Functions import *
import matplotlib.pyplot as plt

import pickle

augmented_data_path = 'augmented_data_CLA_paradigm.pkl'
original_data_path = 'data_CLA_paradigm.pkl'

validation_data_path = 'test_data_CLA.pkl'

auto_encoder_path = 'results_section_24_dim.h5'
encoder = load_encoder(auto_encoder_path)

X_augmented = preprocess_data(augmented_data_path)
X_original = preprocess_data(original_data_path)
X_validation = preprocess_data(validation_data_path)
X_validation = np.expand_dims(X_validation,axis=3)

X_training = np.concatenate((X_augmented,X_original))

BATCH_SIZE = 32
lr = 1e-5
NUM_EPOCHS = 10

train_data = tf.data.Dataset.from_tensor_slices((X_training)).batch(batch_size=BATCH_SIZE).shuffle(buffer_size=len(X_training))

loss = tf.keras.losses.KLDivergence(tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

X_training = np.expand_dims(X_training,axis=3)
initial_embeddings = encoder(X_training)
cluster_centers = initialize_cluster_centers(initial_embeddings)
alpha = 1.

training_loss = []
validation_loss = []

num_steps_per_epoch = len(X_training)//BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    for step,x_batch in enumerate(train_data):

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(cluster_centers)
            current_embeddings = encoder(x_batch)

            S = soft_assignments(current_embeddings,cluster_centers,alpha)
            P = target_distribution(S,cluster_centers)

            KL = loss(S,P)

            grad_Mu = tape.gradient(KL,cluster_centers)
            grad_theta = tape.gradient(KL,encoder.trainable_weights)

            cluster_centers -= lr * grad_Mu
            optimizer.apply_gradients(zip(grad_theta,encoder.trainable_weights))

        if step % 200 == 0:
            print(f'Epoch {epoch} - step {step} Training Loss: {KL.numpy()}, on -log scale: {10/-np.log(KL.numpy())}')

    training_loss.append(10/-np.log(KL.numpy()))

    validation_embeddings = encoder(X_validation)
    S_val = soft_assignments(validation_embeddings,cluster_centers,alpha)
    P_val = target_distribution(S_val,cluster_centers)

    KL_val = loss(S_val,P_val)

    validation_loss.append(10/-np.log(KL_val.numpy()))

    print(f'Epoch {epoch} Validation Loss: {KL_val.numpy()}, On -log scale: {10/-np.log(KL_val.numpy())}')

epochs = np.linspace(start=0,stop=len(training_loss),num=len(training_loss))

try:
    plt.plot(epochs,training_loss,)
    plt.plot(epochs,validation_loss,c='g')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (-10/log(KL Divergence))')

    plt.legend(['Training','Validation'])

    plt.show()
except:
    print('continuing')

encoder.save('DEC_results_dim_24_epochs_equal_10.h5')

with open('cluster_centers_results_24_dim_epochs_equal_10.pkl','wb') as pickle_file:
    pickle.dump(cluster_centers,pickle_file)













