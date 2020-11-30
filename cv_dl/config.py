import os
import imutils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-S", "--stairs", required=False, action = "store_true")
ap.add_argument("-D", "--door", required=False, action = "store_true")
args = ap.parse_args()

# network training parameters
############################################################

# default to sidewalks
ORIG_INPUT_DATASET = "images/sidewalks"

BASE_PATH = "sidewalks"

CLASSES = ("noSidewalk", "sidewalk")
CLASS_NAMES = ("noSidewalk", "sidewalk")


MODEL_PATH = "sidewalk_less_images.model"

AUDIO_PATH = "audio/sidewalk_ahead.m4a"

MODE = "sidewalk"


if args.stairs:
	print("STAIRS MODE")
	MODE = "stairs"
	RIG_INPUT_DATASET = "images/stairs"
	BASE_PATH = "stairs"
	CLASSES = ("noStairs", "stairs")
	CLASS_NAMES = ("noStairs", "stairs")
	MODEL_PATH = "stairs_or_not.model"
	AUDIO_PATH = "audio/stairs.m4a"
elif args.door:
	print("DOOR MODE")
	MODE = "door"
	ORIG_INPUT_DATASET = "images/doors"
	BASE_PATH = "doors"
	CLASSES = ("noDoor, door")
	MODEL_PATH = "doors_or_not.model"
	AUDIO_PATH = "audio/door.m4a"
else:
	print("SIDEWALK MODE")

TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

TEST_IMAGE_PATH = "examples/sidewalk7.jpg"

TRAIN_SPLIT = 0.7

VAL_SPLIT = 0.2


INITIAL_LR = 1e-4
BS = 32
EPOCHS = 20

PLOT_PATH = "plot.png"

# object detection parameters
############################################################

WIDTH = 600
P_SCALE = 1.5
W_STEP = 16
ROI_SIZE = (200, 200)
INPUT_SIZE = (224, 224)


# computer vision parameters
############################################################

NUM_CV_LINES = 20

ACCEPTABLE_ANGLE_DIFF = 0.1

ACCEPTABLE_ENDPOINT_DIST = 150

ACCEPTABLE_PERSPECTIVE_DIFF = 5

