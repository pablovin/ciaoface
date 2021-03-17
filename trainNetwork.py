from Datasets import getGenerator, datasets
from Models import getEncoders, getClassifiers
from Models.classifiers import TRAININGSETUP
import configurations
from datetime import datetime
import os

"""Experiment Parameters"""
datasetName = datasets.DATASET["FER"]
encoderName = getEncoders.ENCODER["VGGFace"]
trainingSetup = TRAININGSETUP["OnlyClassifier"]
withCIAO = True

timeNow = str(datetime.now())
experimentFolder = "/home/pablo/Documents/Datasets/transferLearning/TrainedModels/Experiment/" + str(datasetName)+"/"+str(encoderName)+"_"+str(trainingSetup)+"/"+timeNow

if not os.path.exists(experimentFolder):
    os.makedirs(experimentFolder)

"""Setup configurations"""
configs = configurations.constructConfig(datasetName, encoderName)

"""Obtaining generators"""
generators = getGenerator.getGenerator(datasetName, configs)

"""Obtaining encoders"""
encoder = getEncoders.getEncoder(encoderName, configs)

"""Obtaining Classifier"""
classifier = getClassifiers.getClassifier(encoder, datasetName, withCIAO, configs)

"""Train Classifiers"""
classifier = getClassifiers.trainClassifiers(classifier,experimentFolder, generators, datasetName, trainingSetup, withCIAO, configs)
