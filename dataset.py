import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from common import SIZE, rgb2ycbcr, normalize_y, denormalize_y

'''
Create fixe sized datasets by cropping images into SIZExSIZE pixel sub images 

Pipeline 
    - convert images into fixed size tensors 
    - apply transformations (normalization) to tensors 
    - save tensors as tfrecords 
'''



def process_to_y(img):
    '''
    Takes a rgb PIL image img. 

    Returns a pair of 
        a list of SIZE/R x SIZE/R y channel lr images numpy array 
        a list of   SIZE x SIZE   y channel hr images numpy array
    '''
    lr_list , hr_list = list() , list()
    h, w = img.size
    for i in range(w//SIZE):
        for j in range(h//SIZE):
            hr_rgb = img.crop( (i*SIZE, j*SIZE, (i+1)*SIZE, (j+1)*SIZE) )
            hr_y   = rgb2ycbcr(hr_rgb)[...,0] 
            hr_y   = Image.fromarray( hr_y )
            lr_y   = hr_y.resize( (SIZE//R, SIZE//R) , Image.BICUBIC )
            lr_list.append( np.array(lr_y, dtype="uint8") )
            hr_list.append( np.array(hr_y, dtype="uint8") )
    return (lr_list, hr_list)

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize(lr_value, hr_value):
    features = {"lr": _bytes_feature(lr_value),
                "hr": _bytes_feature(hr_value)}       
    features = tf.train.Features(feature=features)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()

def deserialize(example):
    image_feature_description = {"lr": tf.io.FixedLenFeature([], tf.string),
                                 "hr": tf.io.FixedLenFeature([], tf.string)}
      
    example = tf.io.parse_single_example(example, image_feature_description)
    lr = tf.io.decode_raw(example["lr"], out_type="uint8")
    hr = tf.io.decode_raw(example["hr"], out_type="uint8")
    shape = [SIZE, SIZE, CHN]
    hr = tf.reshape(hr, shape=shape)
    shape = [SIZE//R, SIZE//R, CHN]
    lr = tf.reshape(lr, shape=shape)
    return lr, hr

def inverse_process_to_y(arr):
    '''
    arr: a numpy array of normalized image y channel
    '''
    denormalize_y(arr)
    pass


dataset_type = "valid"
batch_size = 16
data_dir = 'data/DIV2K/DIV2K_' + dataset_type + '_HR/'
tfrecords_dir = "tfrecords/"
files = os.listdir(data_dir)
CHN = 1 # num channels
R = 2   # upscaling facter

record_prefix = "div2k_" + dataset_type
lr_list, hr_list = [], []
os.makedirs(tfrecords_dir, exist_ok=True)

# create writers to save tfrecords 
N_TFRECORDS = 32
writers = []
for i in range(N_TFRECORDS):
    w = tf.io.TFRecordWriter( f"{tfrecords_dir}{record_prefix}{i}.tfrecords" )
    writers.append(w)

sample_count = 0 
for f in files: 
    img = Image.open(data_dir + f)
    # list of numpy arrays of fixed size image y-channels 
    (lr_list, hr_list) = process_to_y(img)
    
    for (lr, hr) in zip(lr_list, hr_list):
        serialized_example = serialize(lr, hr)
        dlr, dhr = deserialize(serialized_example)
        dhr = tf.cast(dhr, dtype="uint8")
        dlr = tf.cast(dlr, dtype="uint8")
        sample_count += 1 
        i = sample_count % N_TFRECORDS
        writers[i].write(serialized_example)  
        sample_count +=1
for i in range(N_TFRECORDS):
    writers[i].close() 
    