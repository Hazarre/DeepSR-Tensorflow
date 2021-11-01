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

batch_size = 16
data_dir = 'data/DIV2K/DIV2K_train_HR/'
tfrecords_dir = "tfrecords/"
files = os.listdir(data_dir)

def _int64_feature(value):
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def process_to_y(img):
    '''
    Takes a rgb PIL image img. 

    Returns a pair of 
        a list of SIZE/4 x SIZE/4 y channel lr images numpy array 
        a list of   SIZE x SIZE   y channel hr images numpy array
    '''
    lr_list , hr_list = list() , list()
    h, w = img.size
    for i in range(w//SIZE):
        for j in range(h//SIZE):
            hr_rgb = img.crop( (i*SIZE, j*SIZE, (i+1)*SIZE, (j+1)*SIZE) )
            hr_y   = rgb2ycbcr(hr_rgb)[...,0] 
            hr_y   = Image.fromarray( hr_y )
            lr_y   = hr_y.resize( (SIZE//4, SIZE//4) , Image.BICUBIC )
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

def serialize(lr_value, hr_value, if_channels=False):
    features = {"lr": _bytes_feature(lr_value),
                "hr": _bytes_feature(hr_value),
                "height": _int64_feature(hr_value.shape[0]),
                "width": _int64_feature(hr_value.shape[1])}
    
    if if_channels:
        features["depth"] = _int64_feature(hr_value.shape[2])
        
    features = tf.train.Features(feature=features)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def deserialize(example, if_channels=False):
    image_feature_description = {"lr": tf.io.FixedLenFeature([], "string"),
                                 "hr": tf.io.FixedLenFeature([], "string"),
                                 "height": tf.io.FixedLenFeature([], "int64"),
                                 "width": tf.io.FixedLenFeature([], "int64")}
    
    if if_channels:
        image_feature_description["depth"] = tf.io.FixedLenFeature([], "int64")
        
    example = tf.io.parse_single_example(example, image_feature_description)
    lr = tf.io.decode_raw(example["lr"], out_type="uint8")
    hr = tf.io.decode_raw(example["hr"], out_type="uint8")
    w , h = example["width"] , example["height"]
    d = example["depth"] if if_channels else 1
    shape = [h, w, d]
    hr = tf.reshape(hr, shape=shape)
    shape = [h//4, w//4, d]
    lr = tf.reshape(lr, shape=shape)
    return lr, hr

def inverse_process_to_y(arr):
    '''
    arr: a numpy array of normalized image y channel
    '''
    denormalize_y(arr)
    pass




record_prefix = "div2k_train"
lr_list, hr_list = [], []
os.makedirs(tfrecords_dir, exist_ok=True)

# create writers to save tfrecords 
n_tfrecords = 32
writers = []
for i in range(n_tfrecords):
    w = tf.io.TFRecordWriter( f"{tfrecords_dir}{record_prefix}{i}.tfrecords" )
    writers.append(w)

sample_count = 0 
for f in files: 
    img = Image.open(data_dir + f)
    # list of numpy arrays of fixed size image y-channels 
    (lr_list, hr_list) = process_to_y(img)
    
    for (lr, hr) in zip(lr_list, hr_list):
        serialized_example = serialize(lr, hr)
        # dlr, dhr = deserialize(serialized_example)
        # dhr = tf.cast(dhr, dtype="uint8")
        # Image.fromarray(dhr.numpy()[...,0]).show()
        # sample_count += 1 
        i = sample_count % n_tfrecords
        writers[i].write(serialized_example)  
        sample_count +=1
for i in range(n_tfrecords):
    writers[i].close() 
    