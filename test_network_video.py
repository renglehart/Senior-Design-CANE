from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from cv_dl import funcs
from cv_dl import config
import numpy as np 
import argparse
import imutils
import time
import cv2
import os
import time

model = load_model(config.MODEL_PATH)

vs = VideoStream(src = 0).start()
# time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width = 500)

	funcs.detect(frame, model)

	# wait 180 seconds, detect_sidewalk is slow and doesn't work in real time
	time.sleep(180)

cv2.destroyAllWindows()
vs.stop()