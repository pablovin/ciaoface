from Models.encoders import getVGGFace, getPK
from Models import trainingParameters

ENCODER = {'VGGFace': 'VGGFace',
             'PK': 'PK'}


def getEncoder(encoder, configs):


    if encoder == ENCODER["VGGFace"]:
        encoder = getVGGFace(configs)
    elif encoder == ENCODER["PK"]:
        encoder = getPK()

    return encoder

