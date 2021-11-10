from tensorflow import keras
from tensorflow.keras.layers import Conv2D, PReLU, Conv2DTranspose
import tensorflow as tf

'''
Correctness of the model: 
The paper counts 12464 parameters excluding PReLU. 
This one has 12637 -- good enough. 
Counting in PReLU, this model had 17.1k params.

From the paper: 
 FSRCNN:     (d=56, s=12, m=4, r=4)
 FSRCNN-s :  (d=32, s=5, m=1, r=4)
'''

class FSRCNN(keras.Model):
    '''
    d: the LR feature dimension
    s: the number of shrinking filters s, and  m
    m: the mapping depth
    r: upscale_factor HR/LR
    '''

    def __init__(self, d=56, s=12, m=4, r=4):
        
        def conv_w_init(shape, dtype=None):
            std = tf.math.sqrt( 2 / (shape[-1] * shape[-2] * shape[-3]))
            return tf.random.normal(shape, mean=0.0, stddev=std)

        super(FSRCNN, self).__init__()
        self.feature_extraction = keras.Sequential([ 
            
            Conv2D(d, (5, 5), (1, 1), "same", 
                    bias_initializer="zeros",
                    kernel_initializer=conv_w_init),
            PReLU(shared_axes=[1,2])
        ], name="feautre_extraction")

        self.shrink = keras.Sequential([ 
            Conv2D(s, (1, 1), (1, 1), "same", 
                    bias_initializer="zeros",
                    kernel_initializer=conv_w_init),
            PReLU(shared_axes=[1,2])
        ], name="shrinking")

        self.map = keras.Sequential(name="mapping")
        for i in range(m):
            self.map.add(
                Conv2D(s, (3, 3), (1, 1), "same",
                        bias_initializer="zeros",
                        kernel_initializer=conv_w_init))
            self.map.add(PReLU(shared_axes=[1,2]))

        self.expand = keras.Sequential([    
            Conv2D(d, (1, 1), (1, 1), "same", 
                    bias_initializer="zeros",
                    kernel_initializer=conv_w_init),
            PReLU(shared_axes=[1,2])
        ], name="expanding")

        self.deconv =  keras.Sequential([
                    Conv2DTranspose(1, (9, 9), 
                     (r, r), padding ="same", output_padding=(r - 1, r -1), kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None))
        ])
    
    def call(self, inputs):
        x = self.feature_extraction(inputs)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x
        
    
if __name__ == "__main__":
    model = FSRCNN()
    model.build((100, None, None, 1))
    model.summary()

