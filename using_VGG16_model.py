from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)

IMAGE_SIZE = [150,150]
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
    
    
    
x = Flatten()(vgg.output)
prediction = Dense(7, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model_mlp.summary()
history = model_mlp.fit(X_train, y_train, epochs =50,verbose = 2,validation_data = [X_test, y_test])


model.evaluate(X_test, y_test) # evaluating the model on the test data



y_preds = model_mlp.predict(X_test)         # predicting the values of the model over test data
y_pred_classes = np.argmax(y_preds,axis = 1)  # selects only the maximum value in each row of the numpy array 
y_true = np.argmax(y_test, axis = 1) 

# calculating confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

#plotting confusion matrix using seaborn
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

#plotting  incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')


#plotting training loss and validation loss curves
plt.plot(history.history['loss'],label = 'Training loss')
plt.plot(history.history['val_loss'],label = 'Validation loss')
plt.title('Training and validation loss curves')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#plotting training loss and validation accuracy curves
plt.plot(history.history['acc'],label = 'Training accuracy')
plt.plot(history.history['val_acc'],label = 'Validation accuracy')
plt.title('Training and validation accuracy curves')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
#printing classification report for our cnn model
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred_classes, target_names=target_names))
