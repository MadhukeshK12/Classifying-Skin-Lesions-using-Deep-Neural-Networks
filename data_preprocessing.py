# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample


tf.random.set_seed(42) # set global random seed


df_skinCancer = pd.read_csv("/HAM10000_metadata.csv") # load metadata.csv as pandas dataframe

print(df_skinCancer.head()) # first five values of the metadata.csv file

# create a dict of image_id as keys and respective path as values

image_path = { os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('/skin_cancer/','*','*.jpg'))}  

# add a path column to the data frame by mapping path to image_id
df_skinCancer['path'] = df_skinCancer['image_id'].map(image_path.get) 

# plot 3 random images of each class from the dataset 

n_samples = 3
fig, m_axs = plt.subplots(7, n_samples, figsize = (5,7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         df_skinCancer.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples,).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
        

le = LabelEncoder()
le.fit(df_skinCancer['dx'])
print(list(le.classes_)) # print the different classes
# transform categorical values into numerical values
df_skinCancer['label'] = le.transform(df_skinCancer['dx']) 

#  resampling the unbalanced data to create balanced data by oversampling minority classes and undersampling majority classes
scale = 500 # resampling size
df_0_balanced = resample(df_0, replace = True, n_samples = scale,random_state = 42)
df_1_balanced = resample(df_1, replace = True, n_samples = scale,random_state = 42)
df_2_balanced = resample(df_2, replace = True, n_samples = scale,random_state = 42)
df_3_balanced = resample(df_3, replace = True, n_samples = scale,random_state = 42)
df_4_balanced = resample(df_4, replace = True, n_samples = scale,random_state = 42)
df_5_balanced = resample(df_5, replace = True, n_samples = scale,random_state = 42)
df_6_balanced = resample(df_6, replace = True, n_samples = scale,random_state = 42)

# concatenate all the resampled classes to create new balanced dataset
df_skin_balanced = pd.concat([df_0_balanced, df_1_balanced, df_2_balanced, df_3_balanced, df_4_balanced, df_5_balanced, df_6_balanced])

# converting input X to numpy array and resizing
# converting output labels to one_hot encoded values
X = np.asarray(df_skin_balanced['image'].tolist())
X = X/255
y = df_skin_balanced['label']
y_one_hot = tf.one_hot(y, depth = 7)
y_one_hot = y_one_hot.numpy()



# splitting dataset into 80% as training data and 20% as test data
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.20, random_state = 42) 

# plotting bar graph for each classes
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
# plotting bar graph of examination techniques
skin_df['dx_type'].value_counts().plot(kind='bar')
# plotting bar graph of localization of lesions on the patient
skin_df['localization'].value_counts().plot(kind='bar')
