import cv2
import numpy
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf


from Datasets.Utils import seq, preProcessFrame, preProcessSequences



"""Generators"""
class generatorImages(Sequence):

    def __init__(self, image_filenames, labels, batch_size, imageSize, augmentation=False, sequence=False, categorical=True, loadURL=True):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.imageSize = (imageSize[0], imageSize[1])

        self.categorical = categorical
        self.loadURL = loadURL

        if imageSize[2]== 1:
            self.grayScale = True
        else:
            self.grayScale = False

        if augmentation:
            self.augmentation = seq
        else:
            self.augmentation = None

        if sequence:
            self.imageSize = (imageSize[1], imageSize[2])
            self.preprocess = preProcessSequences
        else:
            self.preprocess = preProcessFrame




    def __len__(self):
        return int(numpy.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augmentation == None:
            batch = numpy.array([
                self.preprocess(file_name, self.grayScale, self.imageSize, self.loadURL)
                for file_name in batch_x])

        else:
            batch = numpy.array([
                self.augmentation.augment_image(self.preprocess(file_name, self.grayScale, self.imageSize, self.loadURL))
                for file_name in batch_x])

        if self.categorical:
            return batch, batch_y

        else:
            arousal = batch_y[:, 0]
            arousal = numpy.asarray(arousal).astype(numpy.float32)

            valence = batch_y[:, 1]
            valence = numpy.asarray(valence).astype(numpy.float32)

            return batch, [arousal, valence]
