from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.object_detection import non_max_suppression
from cv_dl import config
from playsound import playsound
import numpy as np 
import imutils
import time
import cv2

# object detection helper functions
############################################################

def sliding_window(image, step, windowSize):
	# imageY = image.shape[0]
	# imageX = image.shape[1]
	# windowX = windowSize[0]
	# windowY = windowSize[1]

	for y in range(0, image.shape[0] - windowSize[0], step):
		for x in range (0, image.shape[1] - windowSize[0], step):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	# wid = image.shape[1]
	# height = image.shape[0]
	# minWid = minSize[1]
	# minHeight = minSize[0]

	yield image

	# loop over image pyramid
	while True:
		# get dimensions of next image
		nextW = int(image.shape[1] / scale)
		image = imutils.resize(image, width = nextW)

		# break when image is too small
		if image.shape[0] < minSize[0] or image.shape[1] < minSize[1]:
			break

	# yield next image
	yield image


# detection helper functions
############################################################

def detect(frame, model):

	start = time.time()
	pyramid = image_pyramid(frame, scale = config.P_SCALE, minSize = config.ROI_SIZE)

	rois = []
	locs = []

	(H, W) = frame.shape[:2]

	for image in pyramid:
		# determine scale factor between original image and current pyramid image
		scale = W / float(image.shape[1])

		# for each layer of pyramid, loop over sliding windows
		for (x, y, roiOrig) in sliding_window(image, config.W_STEP, config.ROI_SIZE):

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


	print("[INFO] finished looping over pyramid/windows ...")

	rois = np.array(rois, dtype="float32")

	print("[INFO] classifying ROIs ...")

	predictions = model.predict(rois)

	print("[INFO] finished classifying ROIs ...")


	config.CLASSES = model.predict(rois)[0]
	labels = config.CLASS_NAMES
	# (noSidewalk, sidewalk) = model.predict(rois)[0]


	labels = {}

	for (i, p) in enumerate(predictions):

		if (config.CLASSES[1] > config.CLASSES[0]):
			box = locs[i]
			prob = config.CLASSES[1]
			label = config.CLASS_NAMES[1]

			L = labels.get(label, [])
			L.append((box, prob))
			labels[label] = L
		

	for label in labels.keys():
		# copy image
		print("[INFO] showing results for '{}'".format(label))
		# clone = frame.copy()

		for (box, prob) in labels[label]:
			(startX, startY, endX, endY) = box
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)



		boxes = np.array([p[0] for p in labels[label]])
		probability = np.array([p[1] for p in labels[label]])
		boxes = non_max_suppression(boxes, probability)

		#loop over all boxes kept after NMS
		for(startX, startY, endX, endY) in boxes:
			# draw bounding box
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

		if boxes != None:
			if config.MODE == "sidewalk":
				cv_sidewalk.detect_cv(frame)
			# else:
			# 	playsound(config.AUDIO_PATH)

		end = time.time()
		print("elapsed time: {0}".format(end - start))
		
		cv2.imshow("Frame", frame)
		cv2.waitKey(1) & 0xFF

	print("success")


# computer vision helper functions
############################################################	

def parallel_or_same_line(line1, line2, accept_range):

	# set acceptable ranges
	low_rangeX = line1[1] - accept_range
	high_rangeX = line1[1] + accept_range
	low_rangeY = line1[2] - accept_range
	high_rangeY = line1[2] + accept_range

	# check if values are in range
	if line2[1] > low_rangeX and line2[1] < high_rangeX:
		if line2[2] > low_rangeY and line2[2] < high_rangeY:
			return False
		else:
			return "x only"
	elif line2[3] > low_rangeX and line2[3] < high_rangeX:
		if line2[4] > low_rangeY and line2[4] < high_rangeY:
			return False
		else:
			return "x only"


	# set acceptable ranges
	low_rangeX = line1[3] - accept_range
	high_rangeX = line1[3] + accept_range
	low_rangeY = line1[4] - accept_range
	high_rangeY = line1[4] + accept_range

	# check if values are in range
	if line2[1] > low_rangeX and line2[1] < high_rangeX:
		if line2[2] > low_rangeY and line2[2] < high_rangeY:
			return False
		else:
			return "x only"
	elif line2[3] > low_rangeX and line2[3] < high_rangeX:
		if line2[4] > low_rangeY and line2[4] < high_rangeY:
			return False
		else:
			return "x only"

	return True

def check_perspective(slope, slope_list, accept_range):
	if slope == 0:
		return 

	opp = slope * -1

	# range to check for opposite direction lines with similar slope - perspective lines
	low_range = opp - accept_range
	high_range = opp + accept_range

	# range of slopes too close together (parallel lines)
	low_slope = slope - accept_range
	high_slope = slope + accept_range

	for i in range(len(slope_list)):
		# print("low: {} high: {} found: {}".format(low_range, high_range, slope_list[i]))
		if slope_list[i] > low_slope and slope_list[i] < high_slope:
			continue
		if slope_list[i] > low_range and slope_list[i] < high_range:
			return i





