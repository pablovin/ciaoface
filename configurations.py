from Datasets import datasets
from Models import getEncoders

def constructConfig(dataset, encoder):

    configs = {}

    configs = addImageSizeParam(dataset, configs)
    configs = addEncoderParams(encoder, configs)
    configs = addClassifierParams(encoder, dataset, configs)

    return configs


"""Dataset-specific parameters"""
def addImageSizeParam(dataset, configs):

    if dataset == datasets.DATASET["AffectNetCat"]:
        imgSize = (96,96,3)
    elif dataset == datasets.DATASET["AffectNetDim"]:
        imgSize = (96, 96, 3)
    elif dataset == datasets.DATASET["FER"]:
        imgSize = (96, 96, 3)
    elif dataset == datasets.DATASET["FERPlus"]:
        imgSize = (96, 96, 3)
    elif dataset == datasets.DATASET["JAFFE"]:
        imgSize = (96, 96, 3)
    elif dataset == datasets.DATASET["EmoReact"]:
        imgSize = (96, 96, 3)

    configs["imgSize"] = imgSize

    return configs

"""EncoderParameters"""

def addEncoderParams(encoder, config):

    if encoder == getEncoders.ENCODER["PK"]:
        config["encoderDirectory"] = "PKDirectory/"
        config["lastConvName"]  = "Encoder_conv4"
        config["preLastConvName"]= "Encoder_conv3"
        config["unitsLastConvLayer"] = 512
        config["lastConvShape"] = (5,5)
        config["lastConvStride"] = 2

    elif encoder == getEncoders.ENCODER["VGGFace"]:
        config["lastConvName"]  = "conv5_3"
        config["preLastConvName"]= "conv5_2"
        config["unitsLastConvLayer"] = 512
        config["lastConvShape"] = (3, 3)
        config["lastConvStride"] = 1

    return config


"""Classifier Parameters"""
def addClassifierParams(encoder, dataset, config):

    if dataset == datasets.DATASET["AffectNetCat"] and encoder == getEncoders.ENCODER["VGGFace"]:
        config["outputSize"] = 8
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 256
        config["optmizer"] = "SGD"
        config["initialLearningRate"] =  0.008
        config["temperature"] = 0.05
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 128


    elif dataset == datasets.DATASET["AffectNetCat"] and encoder == getEncoders.ENCODER["PK"]:
        config["outputSize"] = 8
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 512
        config["optmizer"] = "SGD"
        config["initialLearningRate"] =  0.005
        config["temperature"] = 0.05
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 128


    if dataset == datasets.DATASET["AffectNetDim"] and encoder == getEncoders.ENCODER["VGGFace"]:

        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 512
        config["optmizer"] = "SGD"
        config["initialLearningRate"] =  0.008
        config["temperature"] = 0.05
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 128


    elif dataset == datasets.DATASET["AffectNetDim"] and encoder == getEncoders.ENCODER["PK"]:
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 512
        config["optmizer"] = "SGD"
        config["initialLearningRate"] =  0.005
        config["temperature"] = 0.05
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 128

    elif dataset == datasets.DATASET["FER"] and encoder == getEncoders.ENCODER["VGGFace"]:

        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 64
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.0015620329853823292
        config["temperature"] = 0.42417689870668324
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 128

    elif dataset == datasets.DATASET["FER"] and encoder == getEncoders.ENCODER["PK"]:
        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 128
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.003
        config["temperature"] = 0.4
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 128

    elif dataset == datasets.DATASET["FERPlus"] and encoder == getEncoders.ENCODER["VGGFace"]:
        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 128
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.00023762806788752604
        config["temperature"] = 0.23255182441856523
        config["momentum"] = 0.11305107308054507
        config["nesterov"] = True
        config["batchSize"] = 64

    elif dataset == datasets.DATASET["FERPlus"] and encoder == getEncoders.ENCODER["PK"]:
        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 128
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.006
        config["temperature"] = 0.23255182441856523
        config["momentum"] = 0.11305107308054507
        config["nesterov"] = True
        config["batchSize"] = 64

    elif dataset == datasets.DATASET["JAFFE"] and encoder == getEncoders.ENCODER["VGGFace"]:

        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 256
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.0015620329853823292
        config["temperature"] =0.42417689870668324
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 64

    elif dataset == datasets.DATASET["JAFFE"] and encoder == getEncoders.ENCODER["PK"]:
        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 128
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.005
        config["temperature"] = 0.4
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 64

    elif dataset == datasets.DATASET["EmoReact"] and encoder == getEncoders.ENCODER["VGGFace"]:

        config["sequenceSize"] = 10
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 256
        config["LSTMLayers"] = 1
        config["LSTMLayerSize"] = 128
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.4
        config["temperature"] =0.42417689870668324
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 64

    elif dataset == datasets.DATASET["EmoReact"] and encoder == getEncoders.ENCODER["PK"]:
        config["sequenceSize"] = 10
        config["outputSize"] = 7
        config["epoches"] = 50
        config["denseLayers"] = 1
        config["denseLayerSize"] = 128
        config["LSTMLayers"] = 1
        config["LSTMLayerSize"] = 256
        config["optmizer"] = "Adam"
        config["initialLearningRate"] = 0.2
        config["temperature"] = 0.4
        config["momentum"] = 0.4
        config["nesterov"] = True
        config["batchSize"] = 64

    return config
