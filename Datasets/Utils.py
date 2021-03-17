import numpy
import cv2
from imgaug import augmenters as iaa
import sys
"""
Utils
"""
def shuffleData(samples, labels):

    idx = numpy.random.choice(samples.shape[0], samples.shape[0], replace=False)
    x = samples[idx, ...]
    y = labels[idx, ...]

    return x, y



"""Image Utils"""
def preProcessFrame(dataLocation, grayScale, imageSize, loadURL=True):

    if loadURL:
        data = cv2.imread(dataLocation)
    else:
        data = dataLocation

    data = numpy.array(cv2.resize(data, imageSize))

    if grayScale:
       data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
       data = numpy.expand_dims(data, axis=2)
    data = numpy.array(data, dtype='float16')
    data = (data / 255.0)

    return data


def preProcessSequences(dataLocations, grayScale, imageSize, loadURL = True):


    images = []
    for dataLocation in dataLocations:

        if loadURL:
            data = cv2.imread(dataLocation)
        else:
            data = dataLocation

        data = numpy.array(cv2.resize(data, imageSize))
        if grayScale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = numpy.expand_dims(data, axis=2)

        data = numpy.array(data, dtype='float16') / 255.0

        images.append(data)

    images = numpy.array(images)
    return images



seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip
    # sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),
    iaa.OneOf([
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),

    ]),
], random_order=True)


"""Dataset Specific Utils"""


def processEmotionFERplus(emotion_raw):
    '''
    Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:
    Majority: return the emotion that has the majority vote, or unknown if the count is too little.
    Probability or Crossentropty: convert the count into probability distribution.abs
    Multi-target: treat all emotion with 30% or more votes as equal.
    '''
    size = len(emotion_raw)
    emotion_unknown = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal)
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size
    sum_part = 0
    count = 0
    valid_emotion = True
    while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
        maxval = max(emotion_raw)
        for i in range(size):
            if emotion_raw[i] == maxval:
                emotion[i] = maxval
                emotion_raw[i] = 0
                sum_part += emotion[i]
                count += 1
                if i >= 8:  # unknown or non-face share same number of max votes
                    valid_emotion = False
                    if sum(emotion) > maxval:  # there have been other emotions ahead of unknown or non-face
                        emotion[i] = 0
                        count -= 1
                    break
    if sum(
            emotion) <= 0.5 * sum_list or count > 3:  # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
        emotion = emotion_unknown  # force setting as unknown

    return [float(i) / sum(emotion) for i in emotion]