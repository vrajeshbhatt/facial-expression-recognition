#!/usr/bin/env python
# coding: utf-8

# <h2 align=center> Facial Expression Recognition with Keras</h2>

#  

# ### Task 1: Import Libraries

# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from livelossplot.tf_keras import PlotLossesCallback

from IPython.display import SVG, Image
from livelossplot import PlotLossesKerasTF
import tensorflow as tf
print("Tensorflow version:", tf.__version__)


# ### Task 2: Plot Sample Image

# In[3]:


utils.datasets.fer.plot_example_images(plt).show()


# In[6]:


for expression in os.listdir('train/'):
    print(str(len(os.listdir('train/'+expression)))+ ' '+ expression + 'images')


# In[ ]:





# ### Task 3: Generate Training and Validation Batches

# In[44]:


img_size = 48
batch_size= 64

datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator= datagen_train.flow_from_directory('train/',target_size=(img_size,img_size),
                                                  color_mode='grayscale',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator= datagen_train.flow_from_directory('test/',target_size=(img_size,img_size),
                                                  color_mode='grayscale',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)


#  

# ### Task 4: Create CNN Model

# ![](model.png)
# Inspired by Goodfellow, I.J., et.al. (2013). Challenged in representation learning: A report of three machine learning contests. *Neural Networks*, 64, 59-63. [doi:10.1016/j.neunet.2014.09.005](https://arxiv.org/pdf/1307.0414.pdf)

# In[45]:


model = Sequential()

#1-Con Layer
model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2-Con Layer
model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3-Con Layer
model.add(Conv2D(256,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#4-Con Layer
model.add(Conv2D(512,(7,7),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7,activation='softmax'))
opt = Adam(lr=0.0005)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

          


# In[ ]:





#  

# ### Task 6: Train and Evaluate Model

# In[46]:


epochs=30
steps_per_epoch=train_generator.n//train_generator.batch_size
validation_steps=validation_generator.n//validation_generator.batch_size

checkpoint=ModelCheckpoint('model_weights.h5',monitor='val_accuracy',
                          save_weights_only=True,mode='max',verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,min_lr=0.00001,mode='auto')

callbacks = [PlotLossesCallback(),checkpoint,reduce_lr]

history = model.fit(x=train_generator,steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks)


#  

# ### Task 7: Represent Model as JSON String

# In[47]:


model_json=model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)


# In[ ]:




