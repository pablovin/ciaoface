from keras import backend as K
from keras.engine.topology import Layer
import numpy

from keras import backend as K

import tensorflow as tf

class CIAO(tf.keras.layers.Layer):

    def __init__(self, shape, **kwargs, ):
        super(CIAO, self).__init__(**kwargs)
        self.shape = shape

    def build(self, input_shape):

        integrationTerm = numpy.full(self.shape, 0.5, dtype="float32")

        self._integrationTerm = tf.Variable(initial_value=integrationTerm, trainable=True)

        super(CIAO, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        m_nc, i_nc = x

        result = (m_nc / (self._integrationTerm + i_nc))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]