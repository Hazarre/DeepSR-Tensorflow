import os 
import sys

from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
from model import FSRCNN

# Allow local import from parent directory
sys.path.insert(0, "..")
from dataset import deserialize
from common import normalize_y

SIZE = 100
CHN  = 1

def preprocess(example): 
    lr, hr = deserialize(example)
    hr = normalize_y(hr)    
    lr = normalize_y(lr)
    return lr, hr


# config and parameters 
IS_FSRCNN_S = True    # is FSRCNN_S or FSRCNN
RESUME = True    # Train from scratch or use previously traind weights 
n_tfrecords = 32
batch_size = 16
epochs = 50

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
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

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
           initial_epoch=40, 
           epochs=epochs, 
           callbacks=callbacks, 
           validation_data=valid_dataset)


def get_latest_model(): 
    model = FSRCNN(d=32, s=5, m=1, r=4) if IS_FSRCNN_S else FSRCNN()
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer, loss=loss_fn)

    name_model = "FSRCNN_S" if IS_FSRCNN_S else "FSRCNN"
    checkpoint_path  = f"checkpoints/"+ name_model+"{epoch:03d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)