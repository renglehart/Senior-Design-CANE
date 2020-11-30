from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.resnet import preprocess_input
from imutils.object_detection import non_max_suppression
from cv_dl import config
from cv_dl import funcs
import numpy as np 
import argparse
import imutils 
import cv2
import time

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required = False, help = "path to trained model")
# ap.add_argument("-i", "--image", required = True, help = "path to input image")
# args = vars(ap.parse_args())

image = cv2.imread(config.TEST_IMAGE_PATH)
orig = image.copy()
orig = imutils.resize(orig, width=config.WIDTH)
(H, W) = orig.shape[:2]

image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

print("[INFO] loading network ...")
model = load_model(config.MODEL_PATH)


(notFound, found) = model.predict(image)[0]
# config.CLASSES = model.predict(image)[0]


# build label
label = config.CLASS_NAMES[1] if found > notFound else config.CLASS_NAMES[0]
prob = found if found < notFound else notFound
label = "{}: {:.2f}%".format(label, prob * 100)

output = imutils.resize(orig, width = 400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Output", output)
cv2.waitKey(0)

start = time.time()
pyramid = funcs.image_pyramid(orig, scale = config.P_SCALE, minSize = config.ROI_SIZE)

rois = []
locs = []

for image in pyramid:
	# determine scale factor between original image and current pyramid image
	scale = W / float(image.shape[1])

	# for each layer of pyramid, loop over sliding windows
	for (x, y, roiOrig) in funcs.sliding_window(image, config.W_STEP, config.ROI_SIZE):

		# scale ROI corrdinates wrt original dimensions
		x = int(x * scale)
		y = int(y * scale)
		w = int(config.ROI_SIZE[0] * scale)
		h = int(config.ROI_SIZE[1] * scale)

		# process region of interest
		roi = cv2.resize(roiOrig, config.INPUT_SIZE)
		roi = img_to_array(roi)
		roi = preprocess_input(roi)

		# update list of ROIS and coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))

		# if args["visualize"] > 0:
		# 	clone = orig.copy()
		# 	cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# 	cv2.inshow("Visualization", clone)
		# 	cv2.imshow("ROI", roiOrig)
		# 	cv2.waitKey(0)

print("[INFO] finished looping over pyramid/windows ...")

rois = np.array(rois, dtype="float32")

print("[INFO] classifying ROIs ...")

predictions = model.predict(rois)

print("[INFO] finished classifying ROIs ...")


(noSidewalk, sidewalk) = model.predict(rois)[0]


labels = {}

for (i, p) in enumerate(predictions):

	if (config.CLASSES[1] > config.CLASSES[0]):
		box = locs[i]
		prob = config.CLASSES[1]
		label = config.CLASS_NAMES[1]

		L = labels.get(label, [])
		L.append((box, prob))
		labels[label] = L
	# else:
	# 	print("[INFO] no sidewalks were detected here ...")

for label in labels.keys():
	# copy image
	print("[INFO] showing results for '{}'".format(label))
	clone = orig.copy()

	for (box, prob) in labels[label]:
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

	# cv2.imshow("Before", clone)
	# cv2.waitKey(0)


	clone = orig.copy()

	boxes = np.array([p[0] for p in labels[label]])
	probability = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, probability)

	#loop over all boxes kept after NMS
	for(startX, startY, endX, endY) in boxes:
		# draw bounding box
		cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


	end = time.time()
	print("elapsed time: {0}".format(end - start))
	
	cv2.imshow("After", clone)
	cv2.waitKey(0)

print("success")
