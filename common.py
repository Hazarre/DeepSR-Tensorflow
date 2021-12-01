import numpy as np
import tensorflow as tf
from skimage.color import rgb2ycbcr, ycbcr2rgb

# Global Variables 
SIZE = 100  # height and width of training samples
CHN = 1     # number of channels of input/output channels 1 for y and 3 for rgb
R = 4       # The upscaling ratio
DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255