import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, PReLU, Conv2DTranspose, Input
from tensorflow.python.keras.models import Model 


'''
Correctness of the model: 
The paper counts 12464 parameters excluding PReLU. 
This one has 12637 -- good enough. 
Counting in PReLU, this model had 17.1k params. 
'''



def conv_w_init(shape, dtype=None):
    std = tf.math.sqrt( 2 / (shape[-1] * shape[-2] * shape[-3]))
    return tf.random.normal(shape, mean=0.0, stddev=std)



def FSRCNN(n_maps=4, upscale_factor=4):
    input = Input(shape=(None, None, 1))
    #feature extraction 
    x = Conv2D(56, (5, 5), (1, 1), "same",
                    bias_initializer="zeros",
                    kernel_initializer=conv_w_init)(input)
    x = PReLU(shared_axes=[1,2])(x)

    # shrinking layer 
    x = Conv2D(12, (1, 1), (1, 1), "same", 
                    bias_initializer="zeros",
                    kernel_initializer=conv_w_init)(x)
    x = PReLU(shared_axes=[1,2])(x)

    # mapping layers
    for i in range(n_maps):
        x = Conv2D(12, (3, 3), (1, 1), "same",
                        bias_initializer="zeros",
                        kernel_initializer=conv_w_init)(x)
        x = PReLU(shared_axes=[1,2])(x)

    # expanding layers 
    Conv2D(56, (1, 1), (1, 1), "same", 
                    bias_initializer="zeros",
                    kernel_initializer=conv_w_init)(x)
    x = PReLU(shared_axes=[1,2])(x)

    # upscaling with deconv 
    output = Conv2DTranspose(1, (9, 9), 
                     (upscale_factor, upscale_factor), padding ="same", output_padding=(upscale_factor - 1, upscale_factor -1), kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None))(x)

    return Model(input, output, name="FSRCNN")
    


model = FSRCNN()
model.summary()

