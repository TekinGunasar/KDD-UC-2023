# KDD-UC-2023
A repo dedicated to reproducing my results submitted for KDD-UC 2023 in "Deep Embedded Clustering for Unsupervised Motor  Imagery Feature Learning


#Original Data set and Dataset Wrangling
All of the original data set files from the dataset produced by Kaya et. al are included within the zipped folder denoted 'Original Data', while the datasets I extract that are further split into augmented, training, and testing data are in the 'Extracted Data' zipped folder

To see code related to dataset wrangling, this is in the data_wrangling.py folder

#Data Quality Checks
To see code related to the figures in the paper related to the validity of augmented data, please view the 'data_quality_check.py' folder, which currently outputs the cross-correlation matrices between pairs of 1000 random samples from the originial and augmented data, and in commented code is what displays the results of the Kolmogorov-Smirnov tests

#Data augmentation
The actual generation of the augmented data set is done so in the file 'augment_data.py'. The function used that actually augments the data and keeps track of augmented labels as well, is in 'data_wrangling.py' and is called 'augment_dataset'

#Auto-Encoder Pre-Training
The convolutional autio-encoder related code for its model definition and training is 'convolutional_auto_encoder.py'

#DEC Training and Implementation
Code related to training the DEC model is in 'DEC_Helper_Functions.py' and 'DEC.py'. In 'DEC.py' the actual training of the model is done, while in the helper functions file, functions defined with creating sets of soft assignments and target distributions are defined. In DEC_implementation we visualize t-SNE Clusters of learned features, and evaluate their accuracy in classifying data through cluster membershio.

#Testing Supervised classifiers
We test the effectiveness of supervised classifiers in 'testing_supervised_classifiers.py'

#Optimal model

The auto-encoder for the optimal model descrbed in the paper that reached 65.2% accuracy is denoted 'results_section_dim_24.h5'.

The Trained encoder that generates feature representations is denoted 'DEC_results_dim_24_epochs_equal_7.h5'. These store the actual trained sequential model and can be used directly once loaded. One can see how they are loaded in the 'DEC_implementation' file

Learned cluster centers are in 'cluster_centers_results_24_dim_epochs_equal_7.h5'

I hope readers have enjoyed my paper and the potential for unsupervised and self-supervised methods for feature learning can be realized to vastly reduce time needed in creating datasets by removing data annatation from the picture. Perhaps not now, but in many developments to come from both me and other authors that are working in this sub-field. 

- YMKPY 

