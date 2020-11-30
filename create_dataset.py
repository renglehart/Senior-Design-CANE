from cv_dl import config
from imutils import paths
import random
import shutil
import os

# find paths to the images, shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# compute training/testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# validation data
i = int(len(imagePaths) * config.VAL_SPLIT)
valPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# define datasets
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]

for (dType, imagePaths, baseOutput) in datasets:
	print("[INFO] building '{}' split".format(dType))

	# if output base output directory DNE, create int
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	for inputPath in imagePaths:
		# extract filenames, labels
		filename = inputPath.split(os.path.sep)[-1]
		label = inputPath.split(os.path.sep)[-2]

		# build path to label directory
		labelPath = os.path.sep.join([baseOutput, label])

		# if label output director DNE, create it
		if not os.path.exists(labelPath):
			print("[INFO] 'creating{}' directory".format(labelPath))
			os.makedirs(labelPath)

		# construct path to destination image, copy image
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)