"""
'python simpleShoeDetector.py filename'
shoe recognition
"""

from __future__ import print_function
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D
#import matplotlib.pyplot as plt
import os.path
from keras.models import model_from_json
import PIL.ImageOps
import sys
import time
from PIL import Image, ImageDraw, ImageFont


# np.random.seed(1337)  # for reproducibility


def predict(fname=None, ffolder=None):
    if ffolder is not None:
        fpath = ffolder + "/" + fname
    else:
        fpath = fname

    if fname == None:
        print("predict() should have one augment")
        return 1

    f_model = './model'
    model_filename = 'cnn_model.json'
    weights_filename = 'cnn_model_weights.hdf5'

    json_string = open(os.path.join(f_model, model_filename)).read()
    model = model_from_json(json_string)
    model.load_weights(os.path.join(f_model, weights_filename))

    # 2. load picture
    im = Image.open(fpath, 'r')

    if im is None:
        print("Can't open", fpath, ".")
        return 1

    # if the window does not meet our desired window size, resize it
    size = (56, 56)
    window = np.asarray(cv2.resize(cv2.imread(fpath), size)).reshape(-1, size[0], size[1], 3)

    # 5. predict certainty
    probability = model.predict(window, batch_size=26, verbose=1)

    if probability > 0.9:
        return True
    else:
        return False


print(predict(ffolder="testImages", fname="1.jpg"))