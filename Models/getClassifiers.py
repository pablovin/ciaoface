from Datasets.datasets import DATASET
from Models.classifiers import getCategoricalClassifier,  getAVClassifier, getBinaryCrossEntropySequential, trainCategorical, trainDimensional, trainSequenceBinaryCrossEntropy


from Models import getEncoders

def getClassifier(encoder, dataset, withCIAO, configs):

    if dataset == DATASET["AffectNetCat"] or dataset == DATASET["JAFFE"] or dataset == DATASET["FER"] or dataset == DATASET["FERPlus"]:
        model = getCategoricalClassifier(encoder, withCIAO , configs)

    elif dataset == DATASET["AffectNetDim"]:
        model = getAVClassifier(encoder, withCIAO , configs)

    elif dataset == DATASET["EmoReact"]:
        model = getBinaryCrossEntropySequential(encoder, withCIAO , configs)

    return model

def trainClassifiers(model, saveFolder, generators, dataset,trainingSetup, withCIAO, config):

    trainGenerator, validationGenerator = generators[0],generators[1]

    if dataset == DATASET["AffectNetCat"] or dataset == DATASET["JAFFE"] or dataset == DATASET["FER"] or dataset == DATASET["FERPlus"]:

        model = trainCategorical(model, saveFolder, trainGenerator, validationGenerator,  trainingSetup, withCIAO, config)

    elif dataset == DATASET["AffectNetDim"]:
        model = trainDimensional(model, saveFolder, trainGenerator, validationGenerator,  trainingSetup, withCIAO, config)

    elif dataset == DATASET["EmoReact"]:
        model = trainSequenceBinaryCrossEntropy(model, saveFolder, trainGenerator, validationGenerator,  trainingSetup, withCIAO, config)

    return model







