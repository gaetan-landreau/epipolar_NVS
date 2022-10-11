import os
import json
import cv2
import numpy as np
import random
import itertools

SHAPENET_ID_MATCHING = {"03001627": "chair", "02958343": "car"}

# Max number sample to encode the pose.
Nsamples = 5000

# ShapeNet constants
Nviews = 36
N_RANGE = 5
N_MAX_TRAIN = 500
N_MAX_TEST = 198

NULL_COLOR = (0, 0, 0)

# Synthia constants
H = 760
W = 1280

