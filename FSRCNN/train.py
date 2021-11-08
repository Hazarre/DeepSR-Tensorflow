import os 
import sys

from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
from model import FSRCNN

# Allow local import from parent directory
# sys.path.insert(0, "..")
# from dataset import deserialize
# from common import normalize_y


# def preprocess(example): 
#     lr, hr = deserialize(example)
#     hr = normalize_y(hr)    
#     lr = normalize_y(lr)
#     return lr, hr

SIZE = 100
CHN  = 1
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



# config and parameters 
IS_FSRCNN_S = False    # is FSRCNN_S or FSRCNN
RESUME = True   # Train from scratch or use previously traind weights 
start_epoch = 37

n_tfrecords = 32
batch_size = 16
epochs = 300

# prep the dataset
train_dir = [f"../tfrecords/div2k_train{i}.tfrecords" for i in range(n_tfrecords)]
valid_dir = [f"../tfrecords/div2k_valid{i}.tfrecords" for i in range(n_tfrecords)]
train_dataset = tf.data.TFRecordDataset(train_dir).map(preprocess).batch(batch_size)
valid_dataset = tf.data.TFRecordDataset(valid_dir).map(preprocess).batch(batch_size)

# prep the model 
model = FSRCNN(d=32, s=5, m=1, r=4) if IS_FSRCNN_S else FSRCNN()
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer, loss=loss_fn)

name_model = "FSRCNN_S" if IS_FSRCNN_S else "FSRCNN"
log_dir = f"logs/{name_model}"       
checkpoint_path  = f"checkpoints/"+ name_model+"{epoch:03d}.ckpt"


if RESUME: 
    # load pre-trained model 
    # print(checkpoint_path.format(epoch=start_epoch-1))
    # checkpoint_dir = os.path.dirname(checkpoint_path.format(epoch=start_epoch-1))
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print(latest)
    model.load_weights("./checkpoints/FSRCNN017.ckpt")

# Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor = "val_loss", 
                                     patience = 3, 
                                     restore_best_weights = True),

    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                        verbose = 1,
                                        monitor="loss",
                                        save_freq="epoch"),

    tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="epoch")
]

history = model.fit(train_dataset, 
           initial_epoch=start_epoch, 
           epochs=epochs, 
           callbacks=callbacks, 
           validation_data=valid_dataset)
