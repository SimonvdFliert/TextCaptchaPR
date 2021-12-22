# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 00:10:52 2021

@author: Xiao
"""


#%%
#import libs
import tensorflow as tf
import time

import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import numpy as np
from PIL import Image

#%%
#Test run for detecting Tensorflow

print("Tensor Flow Version", {tf.__version__})
print("Keras Version", {tensorflow.keras.__version__})
print("Python", {sys.version})
print("Pandas", {pd.__version__})
print("Scikit-Learn", {sk.__version__})

#Check if TF can detect the GPU
# print(tf.test.gpu_device_name())


#%% TFDataSetBenchMark for testing TF. Download MNIST Fashion dataset, trains an ANN model and test the model


def TFDataSetBenchMark(epoch= 10):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_set_count = len(train_labels)
    test_set_count = len(test_labels)
    
    #setup start time
    t0 = time.time()
    
    #normalize images
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    #create ML model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    #compile ML model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    #train ML model
    model.fit(train_images, train_labels, epochs=epoch)
    
    #evaluate ML model on test set
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    
    #setup stop time
    t1 = time.time()
    total_time = t1-t0
    
    #print results
    print('\n')
    print(f'Training set contained {train_set_count} images')
    print(f'Testing set contained {test_set_count} images')
    print(f'Model achieved {test_acc:.2f} testing accuracy')
    print(f'Training and testing took {total_time:.2f} seconds')




#%%


#%%
#uncomment to test TF and Benchmark your GPU/CPU speed
TFDataSetBenchMark()

#%%

#TODO
#Load each image from the folder
#add label, the file title
#detect the distinct characters from the images


filepath = r"..../Github/TextCaptchaPR/Captcha/captchaImages"
img = Image.open(open(filepath, 'rb'))

image_array = tf.keras.preprocessing.image.img_to_array(img)
print(image_array.shape)
print(image_array)



#%%
