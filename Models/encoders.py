from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten
import tensorflow as tf

from Models import trainingParameters

"""VGGFace"""
def getVGGFace(configs):

    vgg_model = VGGFace(include_top=False,  input_shape=configs["imgSize"])
    last_layer = vgg_model.get_layer('conv5_3').output
    model = Model(inputs=vgg_model.input, outputs=last_layer)

    print ("---------Encoder VGGFace---------")
    model.summary()
    print("---------Encoder VGGFace ---------")

    return model

"""PK"""

def getPK():

    pk = tf.keras.models.load_model(trainingParameters.getPK())
    x = pk.get_layer(name="Encoder_conv4").output
    model = Model(inputs=pk.input, outputs=x)

    print("---------Encoder PK ---------")
    model.summary()
    print("---------Encoder PK ---------")


    return model