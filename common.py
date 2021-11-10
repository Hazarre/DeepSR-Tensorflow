import numpy as np
import tensorflow as tf
from skimage.color import rgb2ycbcr, ycbcr2rgb

# Global Variables 
SIZE = 100  # height and width of training samples
CHN = 1     # number of channels of input/output channels 1 for y and 3 for rgb
R = 4       # The upscaling ratio


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

'''
Data processing, normalization 
'''

def normalize_y(y_array, scale=219, offset=16):
    # for y-channel to [0, 1]
    y_array = tf.cast(y_array, dtype="float32")
    return (y_array - offset) / scale

def denormalize_y(y_array, scale=219, offset=16):
    y_array = y_array * scale + offset
    y_array = tf.clip_by_value(y_array,
                               clip_value_min=offset, 
                               clip_value_max=offset + scale)
    return round_y(y_array)




'''
for srgan and esrgan
'''
def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
