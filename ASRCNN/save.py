import os 
import sys

from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
from model import ASRCNN

SIZE = 100
CHN  = 1
R = 4

IS_ASRCNN_S = False    # is ASRCNN_S or ASRCNN
model = ASRCNN(d=32, s=5, m=1, r=4) if IS_ASRCNN_S else ASRCNN()
latest = tf.train.latest_checkpoint('checkpoints')
model.load_weights(latest)
print("model loaded")



print("model saved")

