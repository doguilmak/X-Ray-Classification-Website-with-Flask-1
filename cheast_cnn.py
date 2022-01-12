"""
Created on Sun Oct 31 01:39:33 2021

@author: doguilmak

dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

"""
#%%
# PART 1 - Importing Libraries

import time
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

#%%
# PART 2 - CNN

# Initialization

#classifier = load_model('model.h5')
#classifier.summary()

classifier = Sequential()
start = time.time()

# Step 1 - First Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))  
# input_shape = (64, 64, 3) size of 64x64 pictures with RGB colors (3 primary colors).

# Step 2 - Pooling 1
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Second Convolution Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

# Step 4 - Pooling 2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 5 - Flattening
classifier.add(Flatten())

# Step 6 - Artificial Neural Network
classifier.add(Dense(output_dim = 128, activation = 'relu'))  # Gives 128bit output
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))  # Returns a value of 1 or 0 (it can be said that a binomial determination is made to determine the male and female class)

# CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  # Gives best probability
classifier.summary()

#%%
# PART 3 - CNN and Pictures

from keras.preprocessing.image import ImageDataGenerator
## ImadeDataGenerator library for pictures.
## The difference from normal picture readings is that it evaluates the pictures one by one, not all at once and helps the RAM to work in a healthy way.

## shear_range = Side bends
## zoom_range = Zoom

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""
val_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""
# Train data
training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')
## target_size= 64x64 size pictures for scan.
## class_mode= Binary set

# Data is tested
test_set = test_datagen.flow_from_directory('chest_xray/train',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')

"""
validation = val_datagen.flow_from_directory(
                                            "chest_xray/val",
                                            target_size=(224, 224),
                                            color_mode="grayscale",
                                            class_mode="binary",
                                            batch_size=32,
                                            shuffle=True,
                                            seed=42,
                                            subset="validation")
"""
# Train Artificial Neural Network

## training_set, = The training set that has been read is added
## nb_epoch = Number of epoch
## samples_per_epoch = How many samples will be made in each epoch
## validation_data = Data to be validated is added
## nb_val_samples = For determine the number of data to be validated 

classifier_history=classifier.fit_generator(training_set,
                         samples_per_epoch = 3000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)

print(classifier_history.history.keys())
print("val_accuracy: ", classifier_history.history['val_accuracy'])
print("accuracy: ", classifier_history.history['accuracy'])
#classifier.save('model.h5')

"""
from keras.utils import plot_model
plot_model(classifier, "binary_input_and_output_model.png", show_shapes=True)
"""

#%%
# PART 4 - Prediction

test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)

## Filter predictions
pred[pred > .5] = 1
pred[pred <= .5] = 0
print('Prediction successful.')

#%%
# PART 5 - Creating Confusion Matrix 

from sklearn.metrics import confusion_matrix

test_labels = []
for i in range(0, int(5216)):  # 5216 samples
    test_labels.extend(np.array(test_set[i][1]))
print('Test Labels(test_labels):\n')
print(test_labels)

# How each file was estimated and compared with real data is shown:
dosyaisimleri = test_set.filenames   
sonuc = pd.DataFrame()
sonuc['dosyaisimleri'] = dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels    

# Confusion matrix
cm = confusion_matrix(test_labels, pred)
print ("Confusion Matrix:\n", cm)

end = time.time()
cal_time = end - start
print("\nTook {} seconds to classificate objects.".format(cal_time))
