import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report
# from cv_dl.letnet import LetNet
# from cv_dl.alexnet import AlexNet
from cv_dl import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#     help = "path to input dataset")
# ap.add_argument("-m", "--model", required=True,
#     help = "path to output model")
# ap.add_argument("-p", "--plot", type = str, default = "plot.png",
#     help = "path to output plot")
# args = vars(ap.parse_args())

totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))


# data augmentation
aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")
valAug = ImageDataGenerator()

mean = np.array([123,68, 116,779, 103.939], dtype="float32")
aug.mean = mean
valAug.mean = mean

trainGen = aug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BS)

valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BS)

testGen = aug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BS)

print("[INFO] preparing model ...")

# initialize model 
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False


print("[INFO] compiling model ...")

# model = AlexNet.build(width = 224, height = 224, depth = 3, classes = 2)
opt = Adam(lr = config.INITIAL_LR, decay = config.INITIAL_LR / config.EPOCHS)

model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

# train network
print("[INFO] training network ...")
history = model.fit(
    trainGen, 
    steps_per_epoch = totalTrain // config.BS, 
    validation_data = valGen, 
    validation_steps = totalVal // config.BS, 
    epochs = config.EPOCHS)

print("[INFO] evaluating network ...")
testGen.reset()


# save model
print("[INFO] saving model ...")
model.save(config.MODEL_PATH, save_format = "h5")

plt.style.use("ggplot")
plt.figure()
N = config.EPOCHS
plt.plot(np.arange(0, N), history.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig(config.PLOT_PATH)
