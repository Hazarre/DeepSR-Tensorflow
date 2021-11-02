from model import FSRCNN

from tensorflow import keras
import tensorflow as tf
from FSRCNN.dataset import deserialize

n_tfrecords = 32
lr 

# prep the model 
model = FSRCNN()
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# prep the dataset
train_dir = [f"tfrecords/train{i}.tfrecords" for i in range(n_tfrecords)]
valid_dir = [f"tfrecords/train{i}.tfrecords" for i in range(n_tfrecords)]

train_ds = tf.data.TFRecordDataset(train_dir).map(deserialize)
valid_ds = tf.data.TFRecordDataset(valid_dir).map(deserialize)

# epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model.compile(optimizer, loss=loss_fn)
model.fit( train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds)

# 91 image dataset for training 
# general 100 image data set for fine tuning 

# learning rate of the convolution layer is 10^-3 
# 10#-4 for deconv layers 
# during fine-tuning the learning rate of all layers is reduced by half. 

