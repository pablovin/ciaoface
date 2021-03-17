from Datasets import datasets
from Datasets.dataLoader import getAffectNetCat, getAffectNetDim, getFER, getFERPlus, getJAFFE, getEmoReact

from Datasets.generators import generatorImages

from Models.trainingParameters import getParamAffectNetCat, getParamAffectNetDim, getParamFER, getParamFERPlus, getParamJAFFE, getParamEmoReact


def getGenerator(generator, configs):

    if generator == datasets.DATASET["AffectNetCat"]:

        imagesTraining, imagesTesting = datasets.getAffectNet()
        trainingX, trainingY = getAffectNetCat(imagesTraining, shuffle=True)
        testingX, testingY = getAffectNetCat(imagesTesting, shuffle=True)

        print("----------------------")
        print( "Training:  X: " + str(trainingX.shape) + " - Y:" + str(trainingY.shape))
        print( "Testing:  X: " + str(testingX.shape) + " - Y:" + str(testingY.shape))
        print("----------------------")

        generatorTraining = generatorImages(trainingX, trainingY, configs["batchSize"], configs["imgSize"], augmentation=False,sequence=False, categorical=True, loadURL=True)
        generatorTesting = generatorImages(testingX, testingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical=True,
                                                 sequence=False, loadURL=True)

        return generatorTraining, generatorTesting

    elif generator == datasets.DATASET["AffectNetDim"]:
        imagesTraining, imagesTesting = datasets.getAffectNet()
        trainingX, trainingY = getAffectNetDim(imagesTraining, shuffle=True)
        testingX, testingY = getAffectNetDim(imagesTesting, shuffle=True)

        print("----------------------")
        print("Training:  X: " + str(trainingX.shape) + " - Y:" + str(trainingY.shape))
        print("Testing:  X: " + str(testingX.shape) + " - Y:" + str(testingY.shape))
        print("----------------------")


        generatorTraining = generatorImages(trainingX, trainingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical=False,
                                                 sequence=False, loadURL=True)
        generatorTesting = generatorImages(testingX, testingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical = False,
                                                sequence=False, loadURL=True)

        return generatorTraining, generatorTesting

    elif generator == datasets.DATASET["FER"]:
        csvFile = datasets.getFER()
        trainingX, trainingY, publicTestingX, publicTestingY, privateTestingX, privateTestingY = getFER(csvFile, shuffle=True)

        print("----------------------")
        print("Training:  X: " + str(trainingX.shape) + " - Y:" + str(trainingY.shape))
        print("Public Testing:  X: " + str(publicTestingX.shape) + " - Y:" + str(publicTestingY.shape))
        print("Private Testing:  X: " + str(privateTestingX.shape) + " - Y:" + str(privateTestingY.shape))
        print("----------------------")


        generatorTraining = generatorImages(trainingX, trainingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical=True,
                                                 sequence=False, loadURL=False)
        generatorPublicTesting = generatorImages(publicTestingX, publicTestingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical = True,
                                                sequence=False, loadURL=False)
        generatorPrivateTesting = generatorImages(publicTestingX, publicTestingY, configs["batchSize"], configs["imgSize"], augmentation=False,
                                                 categorical=True,
                                                 sequence=False, loadURL=False)

        return generatorTraining, generatorPublicTesting, generatorPrivateTesting

    elif generator == datasets.DATASET["FERPlus"]:
        csvFile, csvFilePlus = datasets.getFERPlus()
        trainingX, trainingY, publicTestingX, publicTestingY, privateTestingX, privateTestingY = getFERPlus(csvFile,csvFilePlus, shuffle=True)

        print("----------------------")
        print("Training:  X: " + str(trainingX.shape) + " - Y:" + str(trainingY.shape))
        print("Public Testing:  X: " + str(publicTestingX.shape) + " - Y:" + str(publicTestingY.shape))
        print("Private Testing:  X: " + str(privateTestingX.shape) + " - Y:" + str(privateTestingY.shape))
        print("----------------------")


        generatorTraining = generatorImages(trainingX, trainingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical=True,
                                                 sequence=False, loadURL=False)
        generatorPublicTesting = generatorImages(publicTestingX, publicTestingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical = True,
                                                sequence=False, loadURL=False)
        generatorPrivateTesting = generatorImages(publicTestingX, publicTestingY, configs["batchSize"], configs["imgSize"], augmentation=False,
                                                 categorical=True,
                                                 sequence=False, loadURL=False)

        return generatorTraining, generatorPublicTesting, generatorPrivateTesting

    elif generator == datasets.DATASET["JAFFE"]:
        framesDirectory = datasets.getJaffe()
        trainingX, trainingY, testingX, testingY = getJAFFE(framesDirectory, shuffle=True)

        print("----------------------")
        print("Training:  X: " + str(trainingX.shape) + " - Y:" + str(trainingY.shape))
        print("Testing:  X: " + str(testingX.shape) + " - Y:" + str(testingY.shape))
        print("----------------------")


        generatorTraining = generatorImages(trainingX, trainingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical=True,
                                                 sequence=False, loadURL=True)
        generatorTesting = generatorImages(testingX, testingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical = True,
                                                sequence=False, loadURL=True)


        return generatorTraining, generatorTesting

    elif generator == datasets.DATASET["EmoReact"]:
        imagesTraining,imagesValidation, imagesTest = datasets.getEmoReact()

        trainingX, trainingY = getEmoReact(imagesTraining,framesInSequence=configs["sequenceSize"] ,shuffle=True)
        testingX, testingY = getEmoReact(imagesTraining, framesInSequence=configs["sequenceSize"] ,shuffle=True)
        validationX, validationY = getEmoReact(imagesTraining, framesInSequence=configs["sequenceSize"] ,shuffle=True)

        print("----------------------")
        print("Training:  X: " + str(trainingX.shape) + " - Y:" + str(trainingY.shape))
        print("Testing:  X: " + str(testingX.shape) + " - Y:" + str(testingY.shape))
        print("Validation:  X: " + str(validationX.shape) + " - Y:" + str(validationY.shape))
        print("----------------------")

        generatorTraining = generatorImages(trainingX, trainingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical=True,
                                                 sequence=True, loadURL=True)
        generatorTesting = generatorImages(testingX, testingY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical = True,
                                                sequence=True, loadURL=True)

        generatorValidation = generatorImages(validationX, validationY, configs["batchSize"], configs["imgSize"], augmentation=False, categorical = True,
                                                sequence=True, loadURL=True)


        return generatorTraining, generatorTesting, generatorValidation
