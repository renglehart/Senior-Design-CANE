from cv_dl import config
from cv_dl import funcs
from mplayer import Player
import numpy as np
import cv2
import os

def detect_cv(image):
	
	image = cv2.imread(config.TEST_IMAGE_PATH)

	copy = image.copy()
	copy2 = image.copy()


	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	image = cv2.GaussianBlur(image, (7, 7), 0)

	edges = cv2.Canny(image, 75, 150)

	lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = 200, maxLineGap=50, minLineLength = 200)

	numLines = 0

	# create array to store slopes
	rows, columns = 5, config.NUM_CV_LINES
	slopes = [[0 for i in range(rows)] for j in range(columns)]

	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

		m = (y2 - y1) / (x2 - x1)

		slopes[numLines][0] = m
		slopes[numLines][1] = x1
		slopes[numLines][2] = y1
		slopes[numLines][3] = x2
		slopes[numLines][4] = y2

		numLines = numLines + 1


	# cv2.imshow("linesEdges", edges)
	# cv2.imshow("linesDetected", copy)
	# cv2.waitKey(0)

	print("number of lines detected: {0}".format(numLines))


	# # check for parallel lines 
	# for i in range(numLines):
	# 	m = slopes[i][0]

	# 	low_range = m - config.ACCEPTABLE_ANGLE_DIFF
	# 	high_range = m + config.ACCEPTABLE_ANGLE_DIFF

	# 	for j in range(numLines):
	# 		# print("i is {} slope is {}".format(i, j))
	# 		if slopes[j][0] > low_range and slopes[j][0] < high_range:
	# 			if i != j and j > i:
	# 				print("match found: {} and {}".format(i, j))
	# 				parallel = funcs.parallel_or_same_line(slopes[i], slopes[j], config.ACCEPTABLE_ENDPOINT_DIST)
	# 				print("parallel was {}".format(parallel))
	# 				if parallel == "True":
	# 					print("ADDING BOX")
	# 					cv2.rectangle(copy2, (slopes[j][1], slopes[j][2]), (slopes[j][3], slopes[j][4]),  (255, 0, 0), cv2.FILLED)

	# check for lines that are in opposite angles like a perspective view of a sidewalk
	for i in range(numLines):
		m = [0 for k in range(numLines)]
		
		for j in range(numLines):
			m[j] = slopes[j][0]

		# print(m)
		print("Calling perspective function")
		perspective = funcs.check_perspective(m[i], m, config.ACCEPTABLE_PERSPECTIVE_DIFF)
		print("slope entered: {} perspective slope: {}".format(i, perspective))
		
		if perspective != None:
			pts = np.array([[slopes[i][1], slopes[i][2]], [slopes[i][3], slopes[i][4]],[slopes[perspective][1], slopes[perspective][2]], [slopes[perspective][3], slopes[perspective][4]]], np.int32)
			pts = pts.reshape((-1, 1, 2))
			cv2.polylines(copy2, [pts],  True, (255, 0, 0), thickness = 3)

	# player = Player()
	# abspath = os.path.join(os.path.dirname(config.AUDIO_PATH), 'asdf.m4a')
	# player.loadfile(abspath)
	# playsound(config.AUDIO_PATH)

	cv2.imshow("linesEdges", edges)
	cv2.imshow("Lines", copy)
	cv2.imshow("Outline", copy2)
	cv2.waitKey(0)

	for i in range(numLines):
		print(slopes[i])


