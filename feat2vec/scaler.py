from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class Scaler(Layer):

    def __init__(self, constraint=None, **kwargs):
        self.constraint=constraint
        super(Scaler, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.scale = self.add_weight(name='scale',
                                     shape=(1,),
                                     initializer='normal',
                                     constraint=self.constraint,
                                     trainable=True)
        super(Scaler, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        output = self.scale * x
        return output
