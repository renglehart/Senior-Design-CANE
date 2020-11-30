from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as back

class AlexNet:
    @staticmethod

    def build(width, height, depth, classes):

        #initiate the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we use channels first
        if back.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # layer 1
        model.add(Conv2D(20, (11, 11), padding="same",
            input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

        # layer 2
        model.add(Conv2D(50, (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

        # layer 3
        model.add(Conv2D(20, (3, 3), padding="same",
            input_shape = inputShape))
        model.add(Activation("relu"))

        # layer 4
        model.add(Conv2D(20, (3, 3), padding="same",
            input_shape = inputShape))
        model.add(Activation("relu"))
        
        # layer 5
        model.add(Conv2D(20, (3, 3), padding="same",
            input_shape = inputShape))
        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))


        # fully connected layers

        # layer 6
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))

        # layer 7
        model.add(Dense(4096))
        model.add(Activation("relu"))

        # layer 8 - softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model 
