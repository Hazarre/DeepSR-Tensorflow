import os 

from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr, ycbcr2rgb

from model import ASRCNN

SIZE = 100
R = 4

def normalize_y(y_array, scale=219, offset=16):
    # for y-channel to [0, 1]
    y_array = tf.cast(y_array, dtype="float32")
    return (y_array - offset) / scale

def preprocess(example):
    image_feature_description = {"lr": tf.io.FixedLenFeature([], tf.string),
                                 "hr": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example, image_feature_description)
    lr = tf.io.decode_raw(example["lr"], out_type="uint8")
    hr = tf.io.decode_raw(example["hr"], out_type="uint8")
    shape = [SIZE, SIZE, CHN]
    hr = tf.reshape(hr, shape=shape)
    shape = [SIZE//R, SIZE//R, CHN]
    lr = tf.reshape(lr, shape=shape)
    hr = normalize_y(hr)    
    lr = normalize_y(lr)
    return lr, hr

# load model 
IS_ASRCNN_S = False 

# prep the model 
model = ASRCNN(d=32, s=5, m=1, r=4) if IS_ASRCNN_S else ASRCNN()
model.build((100, None, None, 1))
name_model = "asrcnn_s" if IS_ASRCNN_S else "asrcnn"
weights_dir = "weights_" + name_model + ".h5"
print(weights_dir)
model.load_weights(weights_dir)

f = "/home/henrychang/GitHub/DeepSR-Tensorflow/data/Set5/baby.png"
hr = Image.open(f)
lr = hr.resize((hr.size[0]//4, hr.size[1]//4), Image.BICUBIC)

yuv = (rgb2ycbcr(lr) - 16 )/ 219
yuv = tf.cast(tf.expand_dims(yuv, axis=0), tf.float32)
y_lr= yuv[...,0][..., None]

y_sr = model.predict(y_lr) * 219 + 16
w, h = yuv.shape[1], yuv.shape[2]
yuv_hr = tf.image.resize(yuv, size=[w*R,h*R], method='bicubic', antialias=True) * 219 + 16

yuv_sr = tf.concat( (y_sr, yuv_hr[...,1][...,None], yuv_hr[...,2][...,None]  ) , axis= -1)[0].numpy()
yuv_sr = np.clip( yuv_sr, 16, 235 )
rgb_sr    = np.clip( ycbcr2rgb(yuv_sr)*255, 0, 255 ).astype(np.uint8)


rgb_sr = Image.fromarray(rgb_sr )
rgb_sr.show()