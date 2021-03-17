
DATASET = {'AffectNetCat': 'AffectNetCat',
             'AffectNetDim': 'AffectNetDim',
             "FER":"FER",
             "FERPlus":"FERPlus",
             "JAFFE":"JAFFE",
             "EmoReact":"EmoReact"}


def getAffectNet():

    imagesTraining = "directory/AffectNetProcessed_Training"
    imagesValidation = "directory/AffectNetProcessed_Validation"

    return imagesTraining, imagesValidation

def getFER():

    csvDirectory = "directory/fer2013.csv"

    return csvDirectory

def getFERPlus():
    csvDirectoryFER = "directory/fer2013.csv"
    csvDirectoryFERPlus = "directory/fer2013new.csv"

    return csvDirectoryFER, csvDirectoryFERPlus

def getJaffe():

    framesDirectory = "directory/jaffedbase"

    return framesDirectory

def getEmoReact():

    imagesTraining = "directory/EmoReact/Frames/Train"
    imagesValidation = "directory/EmoReact/Frames/Validation"
    imagesTest = "directory/EmoReact/Frames/Test"

    return imagesTraining,imagesValidation, imagesTest
