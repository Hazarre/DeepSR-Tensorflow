
traintag = 'model6_20210929'

import pickle
import numpy as np
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, initializers, regularizers


initializer = initializers.GlorotUniform()
regularizer = regularizers.L2(1e-7)

inputs = layers.Input(shape=[None,None,1], name='input')
outputs = layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name='Conv-1')(inputs)
outputs = layers.PReLU(shared_axes=[1,2], name='PReLU-1')(outputs)
outputs_0 = outputs

denseblock = dict()

for j in range(1,6):
    k = 2 ** (6 - j)
    denseblock[f'conv_{j}.1'] = layers.Conv2D(k, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'Conv-{j}.1')
    denseblock[f'prelu_{j}.1'] = layers.PReLU(shared_axes=[1,2], name=f'PReLU-{j}.1')
    denseblock[f'concat_{j}.1'] = layers.Concatenate(name=f'Concat-{j}.1')
    denseblock[f'conv_{j}.2'] = layers.Conv2D(k, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'Conv-{j}.2')
    denseblock[f'prelu_{j}.2'] = layers.PReLU(shared_axes=[1,2], name=f'PReLU-{j}.2')
    denseblock[f'concat_{j}.2'] = layers.Concatenate(name=f'Concat-{j}.2')
    denseblock[f'conv_{j}.3'] = layers.Conv2D(k, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'Conv-{j}.3')
    denseblock[f'prelu_{j}.3'] = layers.PReLU(shared_axes=[1,2], name=f'PReLU-{j}.3')
    denseblock[f'concat_{j}.3'] = layers.Concatenate(name=f'Concat-{j}.3')
    denseblock[f'conv_{j}.4'] = layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name=f'Conv-{j}.4')
    denseblock[f'add_{j}'] = layers.Add(name=f'Add-{j}')

concat = layers.Concatenate(name='Concat')
conv_2 = layers.Conv2D(64, 1, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name='Conv-2')
prelu_2 = layers.PReLU(shared_axes=[1,2], name='PReLU-2')
conv_3 = layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name='Conv-3')
prelu_3 = layers.PReLU(shared_axes=[1,2], name='PReLU-3')
add = layers.Add(name='Add')

for i in range(1,4):
    temp = list()
    for j in range(1,6):
        temp_1 = denseblock[f'conv_{j}.1'](outputs)
        temp_1 = denseblock[f'prelu_{j}.1'](temp_1)
        temp_1 = denseblock[f'concat_{j}.1']([temp_1, outputs])
        temp_2 = denseblock[f'conv_{j}.2'](temp_1)
        temp_2 = denseblock[f'prelu_{j}.2'](temp_2)
        temp_2 = denseblock[f'concat_{j}.2']([temp_2, temp_1])
        temp_3 = denseblock[f'conv_{j}.3'](temp_2)
        temp_3 = denseblock[f'prelu_{j}.3'](temp_3)
        temp_3 = denseblock[f'concat_{j}.3']([temp_3, temp_2])
        temp_3 = denseblock[f'conv_{j}.4'](temp_3)
        temp_3 = denseblock[f'add_{j}']([outputs, temp_3])
        temp.append(temp_3)

    temp.append(outputs_0)

    outputs = concat(temp)
    outputs = conv_2(outputs)
    outputs = prelu_2(outputs)
    outputs = conv_3(outputs)
    outputs = prelu_3(outputs)
    outputs = add([outputs, outputs_0])
    
outputs = layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name='Deconv-A')(outputs)
outputs = layers.PReLU(shared_axes=[1,2], name='PReLU-A')(outputs)
outputs = layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer, name='Deconv-B')(outputs)
outputs = layers.PReLU(shared_axes=[1,2], name='PReLU-B')(outputs)
outputs = layers.Conv2D(1, 1, padding='same', kernel_initializer=initializer, name='Conv-4')(outputs)
outputs = layers.PReLU(shared_axes=[1,2], name='PReLU-4')(outputs)

mcdn = keras.Model(inputs, outputs)