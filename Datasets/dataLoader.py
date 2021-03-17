import tqdm
import os
import numpy
import tensorflow as tf
import csv
from tqdm import tqdm

from Datasets.Utils import shuffleData, processEmotionFERplus


"""AffectNet"""

def getAffectNetCat(framesDirectory, shuffle):
    samples = []
    labels = []

    for fileName in tqdm(os.listdir(framesDirectory)):
        splitName = fileName.split("__")
        category = splitName[1]
        # print ("category:" + str(category))
        if int(category) <=7:
            samples.append(framesDirectory + "/" + fileName)
            labels.append(int(category))

    labels = tf.keras.utils.to_categorical(labels, 8, dtype="float32")
    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples, labels)


    return samples, labels

def getAffectNetDim(framesDirectory, shuffle):

    samples = []
    labels = []

    for fileName in tqdm(os.listdir(framesDirectory)):
        splitName = fileName.split("__")

        arousal = splitName[2]
        valence = splitName[3][0:-4]

        # print ("Split name:" + str(splitName[3][0:-4]))
        # input("here")
        samples.append(framesDirectory+"/"+fileName)
        labels.append([float(arousal), float(valence)])

    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels


"""FER/FER+"""
def getFER(csvDirectory, shuffle):
    samplesTraining = []
    labelsTraining = []

    samplesPublicTesting = []
    labelsPublicTesting = []

    samplesPrivateTesting = []
    labelPrivateTesting = []

    classes = []
    with open(csvDirectory) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line_count, row in tqdm(enumerate(csv_reader)):
            if line_count> 0 and len(row)>0 :
                if not int(row[0]) == 2:
                    image = row[1].split(" ")
                    # print ("Image:" + str(image))
                    image = [float(int(number) / 255) for number in image]
                    # print ("ROw:" + str(row))
                    image = numpy.reshape(image, (48,48))
                    image = [image, image, image]
                    image = numpy.reshape(image, (48, 48, 3))
                    # label = [float(int(number)/numpy.sum(labelAll)) for number in labelAll]

                    if row[2] == "Training":
                        labelsTraining.append(int(row[0]))
                        samplesTraining.append(image)
                    if not int(row[0]) in classes:
                        classes.append(int(row[0]))

                    elif row[2] == "PublicTest":
                        labelsPublicTesting.append(int(row[0]))
                        samplesPublicTesting.append(image)

                    else:
                        labelPrivateTesting.append(int(row[0]))
                        samplesPrivateTesting.append(image)

    samplesTraining, labelsTraining = numpy.array(samplesTraining), numpy.array(labelsTraining)
    samplesPublicTesting, labelsPublicTesting = numpy.array(samplesPublicTesting), numpy.array(labelsPublicTesting)
    samplesPrivateTesting, labelPrivateTesting = numpy.array(samplesPrivateTesting), numpy.array(labelPrivateTesting)

    labelsTraining =  tf.keras.utils.to_categorical(labelsTraining, 7, dtype="float32")
    labelsPublicTesting = tf.keras.utils.to_categorical(labelsPublicTesting, 7, dtype="float32")
    labelPrivateTesting = tf.keras.utils.to_categorical(labelPrivateTesting, 7, dtype="float32")

    if shuffle:
        samplesTraining, labelsTraining = shuffleData(samplesTraining, labelsTraining)
        samplesPublicTesting, labelsPublicTesting = shuffleData(samplesPublicTesting, labelsPublicTesting)
        samplesPrivateTesting, labelPrivateTesting = shuffleData(samplesPrivateTesting, labelPrivateTesting)

    return samplesTraining, labelsTraining, samplesPublicTesting, labelsPublicTesting, samplesPrivateTesting, labelPrivateTesting


def getFERPlus(csvDirectory, csvDirectoryPlus, shuffle):
    samplesTraining = []
    labelsTraining = []

    samplesPublicTesting = []
    labelsPublicTesting = []

    samplesPrivateTesting = []
    labelPrivateTesting = []

    with open(csvDirectoryPlus) as csvfile2:
        readCSV = list(csv.reader(csvfile2, delimiter=','))

        with open(csvDirectory) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line_count, row in tqdm(enumerate(csv_reader)):
                if line_count> 0 and len(row)>0 :
                    image = row[1].split(" ")
                    # print ("Image:" + str(image))
                    image = [float(int(number) / 255) for number in image]
                    # print ("ROw:" + str(row))
                    image = numpy.reshape(image, (48,48))
                    image = [image, image, image]
                    image = numpy.reshape(image, (48, 48, 3))
                    # print ("Shape:" + str(numpy.array(image).shape))
                    # input("here")
                    rowLabel = readCSV[line_count]
                    label = rowLabel[2:-1]
                    # print ("Label:" + str(label))
                    labelAll = [int(number) for number in label]
                    # print("label before: " + str(labelAll))

                    label = processEmotionFERplus(labelAll)
                    # print ("label after:" + str(label))

                    idx = numpy.argmax(label)
                    if idx < 7:  # not unknown or non-face
                        label = label[:-2]
                        label = [float(i) / sum(label) for i in label]


                        if row[2] == "Training":
                            labelsTraining.append(label)
                            samplesTraining.append(image)

                        elif row[2] == "PublicTest":
                            labelsPublicTesting.append(label)
                            samplesPublicTesting.append(image)

                        else:
                            labelPrivateTesting.append(label)
                            samplesPrivateTesting.append(image)

    samplesTraining, labelsTraining = numpy.array(samplesTraining), numpy.array(labelsTraining)
    samplesPublicTesting, labelsPublicTesting = numpy.array(samplesPublicTesting), numpy.array(labelsPublicTesting)
    samplesPrivateTesting, labelPrivateTesting = numpy.array(samplesPrivateTesting), numpy.array(labelPrivateTesting)

    labelsTraining = tf.keras.utils.to_categorical(labelsTraining, 7, dtype="float8")
    labelsPublicTesting = tf.keras.utils.to_categorical(labelsPublicTesting, 7, dtype="float8")
    labelPrivateTesting = tf.keras.utils.to_categorical(labelPrivateTesting, 7, dtype="float8")

    if shuffle:
        samplesTraining, labelsTraining = shuffleData(samplesTraining, labelsTraining)
        samplesPublicTesting, labelsPublicTesting = shuffleData(samplesPublicTesting, labelsPublicTesting)
        samplesPrivateTesting, labelPrivateTesting = shuffleData(samplesPrivateTesting, labelPrivateTesting)

    return samplesTraining, labelsTraining, samplesPublicTesting, labelsPublicTesting, samplesPrivateTesting, labelPrivateTesting

"""JAFFE"""

def getJAFFE(imagesDirectory, shuffle):

    labels = {"AN":0, "DI":1, "FE":2, "HA":3, "NE":4, "SA":5, "SU":6}

    samplesTraining = []
    labelsTraining = []
    samplesTesting = []
    labelsTesting = []

    classesIntest = []
    for a in range(len(labels)):
        classesIntest.append(0)

    imgs = os.listdir(imagesDirectory)

    for img in imgs:
        label = img.split(".")[1][0:2]
        labelIndex = int(labels[label])
        if classesIntest[labelIndex] < 10:
            classesIntest[labelIndex] += 1
            samplesTesting.append(imagesDirectory+"/"+img)
            labelsTesting.append(labelIndex)
        else:
            labelsTraining.append(labelIndex)
            samplesTraining.append(imagesDirectory+"/"+img)


    labelsTraining = tf.keras.utils.to_categorical(labelsTraining, len(labels), dtype="float32")
    labelsTesting = tf.keras.utils.to_categorical(labelsTesting,  len(labels), dtype="float32")

    samplesTesting, labelsTesting = numpy.array(samplesTesting), numpy.array(labelsTesting)
    samplesTraining, labelsTraining = numpy.array(samplesTraining), numpy.array(labelsTraining)


    if shuffle:
        samplesTesting, labelsTesting = shuffleData(samplesTesting, labelsTesting)
        samplesTraining, labelsTraining = shuffleData(samplesTraining, labelsTraining)

    return samplesTraining, labelsTraining, samplesTesting  , labelsTesting

"""EmoReact"""


def getEmoReact(videoDirectory, shuffle, framesInSequence):

    """
    1- Curiosity
    2- Uncertainty
    3- Excitement
    4- Happiness
    5- Surprise
    6- Disgust
    7- Fear
    8- Frustration
    9- Valence
    """
    samples = []
    labels = []
    for video in tqdm(os.listdir(videoDirectory)):
        framesNames = os.listdir(videoDirectory+"/"+video+"/")
        framesNames = sorted(framesNames, key=lambda x: int(x.split("_")[0]))
        currentFrameList = []
        for fileName in framesNames:
            splitName = fileName.split("_")[1].split(",")
            #0_0,0,0,0,0,0,0,0,2.6667
            classes = splitName[0:8]
            # print ("Classes:" + str(classes))
            # valence = float(int(splitName[8])/7)
            # print ("len:"  + str(len(classes)))
            if len(classes) == 8:
                label = [0.0]*len(classes)
                for indexV, value in enumerate(classes):
                    if value == "1":
                        label[indexV] = 1
                currentFrameList.append(videoDirectory+"/"+video+"/"+"/"+fileName)
                if len(currentFrameList)==framesInSequence:
                    samples.append(currentFrameList)
                    labels.append(label)
                    currentFrameList = []

    samples, labels = numpy.array(samples), numpy.array(labels)

    # print ("Samples:" + str(samples.shape))
    # print("Labels:" + str(labels.shape))
    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels

