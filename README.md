# Classifying Skin Lesions using Deep Neural Networks

## Abstract: 
In the human body, cells typically go through a set process in order to reproduce. To ensure appropriate body function, its genesis, active period, and death should occur in the proper order. The disrupt in this order causes cancer. In the UK, one of the most prevalent cancers is skin cancer. In the UK, more than 16,000 new cases of melanoma skin cancer are detected each year, ranking it as the fifth most prevalent cancer overall, according to Cancer Research UK. Artificial neural networks are utilised to detect and learn features that are already present in the images. With the help of the HAM10000 ("Human Against Machine with 10015 training images") dataset, we train our model to classify these skin lesion. Here, we use a convolution neural network with the Keras TensorFlow API to train our model to recognise seven different types of skin cancer. In order to classify our skin lesions, we first used a multilayer perceptron to train our model, which has a validation accuracy of 70.57%. We then used a convolution neural network (CNN) to train our second model, which has a validation accuracy of 75.20%. Finally, we apply transfer learning by using VGGNET 16 with pre-trained weights over the ImageNet database, and achieved an accuracy of 72.29 %.
# Dataset:
The dataset was downloaded from Kaggle, originally the dataset HAM10000 was a part of ISIC 2018 classification challenge. This collection comprises of dermatoscopic images from various populations that were collected and saved using various modalities. It comprises of 10015 dermatoscopic images that are made available as a training set for scholarly machine learning projects and are accessible to the general public via the ISIC repository. 

There are seven distinct categories of skin cancer, as follows:

![image](https://github.com/MadhukeshK12/skin_lesion_classifier/assets/115413028/c7f78f1a-abb9-4464-b735-ccbb1b09bd8e)


Dataset link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


load the data_preprocessing.py before running the models
