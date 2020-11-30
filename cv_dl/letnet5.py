import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class LetNet5:
	@staticmethod

	def build(width, height, depth, classes):
		### Construct LeNet-5 and Print out Model Summary
		model = Sequential()
        inputShape = (height, width, depth)

        # if we use channels first
        if back.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
		# First layer

		# Convolution layer 1
		model.add(tf.keras.layers.Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (32,32,1)))

		# Pooling layer 1
		model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))


		# Second layer

		# Convolution layer 2
		model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (14, 14, 6)))

		# Pooling layer 2
		model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2))

		# Flatten
		model.add(tf.keras.layers.Flatten())


		# Third layer

		# First fully connected layer
		model.add(tf.keras.layers.Dense(units = 120, activation = 'relu'))


		# Fourth layer

		# Second fully connected layer
		model.add(tf.keras.layers.Dense(units = 84, activation = 'relu'))

		# Fifth layer

		#Output layer
		model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))

		return model

