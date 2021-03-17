from keras.layers import Dense, Dropout, Flatten, Lambda, TimeDistributed, LSTM, MaxPooling2D, Conv2D, InputLayer
from keras.models import Input
from Models.CIAO import CIAO
from keras.models import Model
from keras.optimizers import Adam, SGD
from Metrics.metrics import ccc
import time

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import keras
import tensorflow_addons as tfa

import os

from Utils.Utils import createFolders

"""Dictionaries"""

TRAININGSETUP = {'All_FromScratch': 'All_FromScratch',
             'OnlyClassifier': 'OnlyClassifier',
             "OnlyLastConv":"OnlyLastConv",
             "All":"All"}


"""CIAO"""
def addCIAO(model, config):

    preLastConv = model.get_layer(config["preLastConvName"]).output
    lastConv = model.get_layer(config["lastConvName"],).output

    if config["lastConvStride"] > 1:
        inhibitionConv = Conv2D(config["unitsLastConvLayer"], config["lastConvShape"], padding="same", kernel_initializer="glorot_uniform",strides=config["lastConvStride"],
                                  activation="relu",
                                  name="Inhibition_CIAO")(preLastConv)
    else:
        inhibitionConv = Conv2D(config["unitsLastConvLayer"], config["lastConvShape"], padding="same", kernel_initializer="glorot_uniform",
                                activation="relu",
                                name="Inhibition_CIAO")(preLastConv)

    inhibition = CIAO(name="CIAO", shape=(1, 1, config["unitsLastConvLayer"]))([lastConv, inhibitionConv])


    return inhibition


"""Utils"""

def setCIAOTrainingSetup(model):
    for layer in model.layers:
        layer.trainable = False

    model.get_layer(name="Inhibition_CIAO").trainable = True
    model.get_layer(name="CIAO").trainable = True

    return model


def setTrainingLayerSetup(model, trainingLayer, lastConvName, denseLayers):

    if trainingLayer == TRAININGSETUP["All_FromScratch"]:
        model = resetLayers(model)
    else:
        if trainingLayer == TRAININGSETUP["OnlyClassifier"] or trainingLayer== TRAININGSETUP['OnlyLastConv']:
            for layer in model.layers:
                layer.trainable = False

        if trainingLayer == TRAININGSETUP["OnlyLastConv"]:
            model.get_layer(name=lastConvName).trainable = True

    for i in range(denseLayers):
        model.get_layer(name="denseLayer" + str(i)).trainable = True

    model.get_layer(name="category_output").trainable = True

    return model

def resetLayers(model):
    # print ("Weight Old:" + str(model.layers[1].get_weights()))
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer

            old_weights, old_biases = model.layers[ix].get_weights()

            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=len(old_biases))])

    return model

def loadTrainedModel(directory, verbose=0):

   model = keras.models.load_model(directory,
                             custom_objects={'ccc': ccc, 'SupervisedContrastiveLoss': SupervisedContrastiveLoss},
                                    compile=False)

   if verbose:
       print ("----------------")
       print ("Loaded:" + str(directory))
       model.summary()
       print ("----------------")

   return model



"""Training"""

class SupervisedContrastiveLoss(keras.losses.Loss):

    CLASSIFIERTYPE = {'categorical': 'categorical',
             'mse': 'mse',
                      "BinaryCrossEntropy": "BinaryCrossEntropy"}


    def __init__(self, temperature=1, classifierType = CLASSIFIERTYPE["categorical"], name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature
        self.classifierType = classifierType

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # print("Shape labels:" + str(labels.shape))
        # print("Shape feature_vectors:" + str(feature_vectors.shape))

        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # print("Shape feature_vectors normalized:" + str(feature_vectors_normalized.shape))
        # print("Shape transpose  normalized:" + str(tf.transpose(feature_vectors_normalized).shape))
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        if self.classifierType == self.CLASSIFIERTYPE["categorical"]:
            labels = tf.math.argmax(labels,1)
        elif self.classifierType == self.CLASSIFIERTYPE["mse"]:
            labels = tf.cast(labels, dtype=tf.float32)

            labels = tf.div(
                tf.subtract(
                    labels,
                    tf.reduce_min(labels)
                ),
                tf.subtract(
                    tf.reduce_max(labels),
                    tf.reduce_min(labels)
                ) )

            labels = tf.cast(tf.math.scalar_mul(10, labels), dtype=tf.int32)


        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


@tf.function
def train_step(x, y, model, optimizer, loss, trainAccuracy):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    trainAccuracy.update_state(y, logits)
    return loss_value

@tf.function
def test_step(x, y, model, validationAccuracy):
    val_logits = model(x, training=False)
    validationAccuracy.update_state(y, val_logits)



def trainSequenceBinaryCrossEntropy(model, saveFolder, trainGenerator, validationGenerator,  trainingSetup, trainWithCIAO, config):


    # Adjust the classifier updatable parameters
    model = setTrainingLayerSetup(model, trainingSetup, config["lastConvName"],config["denseLayers"] )

    print("----------------")
    print("Training classifier model:")
    model.summary()
    print("----------------")


    if trainWithCIAO:
        # Adjust the CIAO updatable parameters
        model = setCIAOTrainingSetup(model)

        print("----------------")
        print("Training CIAO model:")
        model.summary()
        print("----------------")

    """Create folders for training"""
    modelFolder = saveFolder+"/Model"
    createFolders(modelFolder)


    """Original model train parameters"""
    # Instantiate the optmizer
    if config["optmizer"] == "SGD":
      optmizer = SGD(config["initialLearningRate"], momentum=config["momentum"], nesterov=config["nesterov"])
    elif config["optmizer"] == "Adam":
        optmizer = Adam(config["initialLearningRate"])

    # Instantiate lossFunctions
    losses = []
    metricsTrain = []
    metricsTest = []
    for a in range(config["outputSize"]):
        losses.append(tf.keras.losses.BinaryCrossentropy(from_logits=True))
        metricsTrain.append(tf.keras.metrics.CategoricalAccuracy())
        metricsTest.append(tf.keras.metrics.CategoricalAccuracy())


    """CIAO model train parameters
    """
    if trainWithCIAO:
        # Instantiate the optmizer
        if config["optmizer"] == "SGD":
            optmizerCIAO = SGD(config["initialLearningRate"], momentum=config["momentum"], nesterov=config["nesterov"])
        elif config["optmizer"] == "Adam":
            optmizerCIAO = Adam(config["initialLearningRate"])

        # Instantiate lossFunctions
        # Instantiate lossFunctions
        lossesCIAO = []
        metricsCIAO = []
        for a in range(config["outputSize"]):
            lossesCIAO.append(SupervisedContrastiveLoss(config["temperature"], SupervisedContrastiveLoss.CLASSIFIERTYPE["BinaryCrossEntropy"]))
            metricsCIAO.append(tf.keras.metrics.CategoricalAccuracy())


    for epoch in range(config["epoches"]):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(trainGenerator):

            totalLoss = 0
            # If train with CIAO, first
            if trainWithCIAO:
                model = setCIAOTrainingSetup(model)

                CIAOLoss = 0
                for a in range(config["outputSize"]):
                    modelArousal = Model(inputs=model.input, outputs=model.get_layer("class" + str(a) + "_output").output)
                    CIAOLoss += train_step(x_batch_train, y_batch_train, modelArousal, optmizerCIAO, lossesCIAO[a], metricsCIAO[a])

                model = setTrainingLayerSetup(model, trainingSetup, config["lastConvName"],config["denseLayers"])

            """Train the original Model"""
            combinedLoss = 0
            for a in range(config["outputSize"]):
                modelArousal = Model(inputs=model.input, outputs=model.get_layer("class" + str(a) + "_output").output)
                combinedLoss += train_step(x_batch_train, y_batch_train, modelArousal, optmizerCIAO, losses[a],
                                       metricsTrain[a])

            # Log every 200 batches.
            if step % 200 == 0:
                if trainWithCIAO:
                    print(
                        "Training loss (for one batch) at step %d: %.4f - CIAO loss:%.4f "
                        % (step, float(combinedLoss), float(CIAOLoss))
                    )
                else:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(combinedLoss))
                    )
                print("Seen so far: %d samples" % ((step + 1) * config["batchSize"]))

        # Display metrics at the end of each epoch.
        for a in range(config["outputSize"]):
            acc =  metricsTrain[a].result()
            print("Accuracy over epoch Class %f:  %.4f " % (float(a),float(acc)))
            metricsTrain[a].reset_states()
        model.save(modelFolder)

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in validationGenerator:
            for a in range(config["outputSize"]):
                modelArousal = Model(inputs=model.input, outputs=model.get_layer("class" + str(a) + "_output").output)
                test_step(x_batch_val, y_batch_val, modelArousal, metricsTest[a])


                testMetric = metricsTest[a].result()

                metricsTest[a].reset_states()
                print("Accuracy over epoch Class %f:  %.4f " % (float(a), float(testMetric)))

        print("Time taken: %.2fs" % (time.time() - start_time))



    return model

def trainDimensional(model, saveFolder, trainGenerator, validationGenerator,  trainingSetup, trainWithCIAO, config):


    # Adjust the classifier updatable parameters
    model = setTrainingLayerSetup(model, trainingSetup, config["lastConvName"],config["denseLayers"] )

    print("----------------")
    print("Training classifier model:")
    model.summary()
    print("----------------")


    if trainWithCIAO:
        # Adjust the CIAO updatable parameters
        model = setCIAOTrainingSetup(model)

        print("----------------")
        print("Training CIAO model:")
        model.summary()
        print("----------------")

    """Create folders for training"""
    modelFolder = saveFolder+"/Model"
    createFolders(modelFolder)


    """Original model train parameters"""
    # Instantiate the optmizer
    if config["optmizer"] == "SGD":
      optmizer = SGD(config["initialLearningRate"], momentum=config["momentum"], nesterov=config["nesterov"])
    elif config["optmizer"] == "Adam":
        optmizer = Adam(config["initialLearningRate"])

    # Instantiate lossFunctions
    lossMSEArousal =  tf.keras.losses.MSE(from_logits=True)
    lossMSEValence = tf.keras.losses.MSE(from_logits=True)

    """CIAO model train parameters
    """
    if trainWithCIAO:
        # Instantiate the optmizer
        if config["optmizer"] == "SGD":
            optmizerCIAO = SGD(config["initialLearningRate"], momentum=config["momentum"], nesterov=config["nesterov"])
        elif config["optmizer"] == "Adam":
            optmizerCIAO = Adam(config["initialLearningRate"])

        # Instantiate lossFunctions
        lossCIAOArousal = SupervisedContrastiveLoss(config["temperature"], SupervisedContrastiveLoss.CLASSIFIERTYPE["mse"])
        lossCIAOVaelence = SupervisedContrastiveLoss(config["temperature"], SupervisedContrastiveLoss.CLASSIFIERTYPE["mse"])


    #Prepare the Metrics
    trainCCCArousal = ccc()
    trainCCCValence = ccc()

    testCCCArousal = ccc()
    testCCCValence = ccc()

    for epoch in range(config["epoches"]):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(trainGenerator):

            totalLoss = 0
            # If train with CIAO, first
            if trainWithCIAO:
                model = setCIAOTrainingSetup(model)

                modelArousal = Model(inputs=model.input, outputs=model.get_layer("arousal_output").output)

                loss_value_CIAO_arousal = train_step(x_batch_train, y_batch_train, modelArousal, optmizerCIAO, lossCIAOArousal, trainCCCArousal)

                modelValence = Model(inputs=model.input, outputs=model.get_layer("valence_output").output)

                loss_value_CIAO_valence = train_step(x_batch_train, y_batch_train, modelValence, optmizerCIAO,
                                                     lossCIAOVaelence, trainCCCValence)

                model = setTrainingLayerSetup(model, trainingSetup, config["lastConvName"],config["denseLayers"])

            """Train the original Model"""
            modelArousal = Model(inputs=model.input, outputs=model.get_layer("arousal_output").output)
            loss_value_arousal = train_step(x_batch_train, y_batch_train, modelArousal, optmizer, lossMSEArousal, trainCCCArousal)

            modelValence = Model(inputs=model.input, outputs=model.get_layer("valence_output").output)
            loss_value_valence = train_step(x_batch_train, y_batch_train, modelValence, optmizer, lossMSEValence,
                                            trainCCCValence)

            # Log every 200 batches.
            if step % 200 == 0:
                if trainWithCIAO:
                    print(
                        "Training loss (for one batch) at step %d: %.4f - CIAO loss:%.4f "
                        % (step, float(loss_value_arousal+loss_value_valence), float(loss_value_CIAO_arousal+loss_value_CIAO_valence))
                    )
                else:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value_arousal+loss_value_valence))
                    )
                print("Seen so far: %d samples" % ((step + 1) * config["batchSize"]))

        # Display metrics at the end of each epoch.
        train_cccArousal = trainCCCArousal.result()
        train_cccValence = trainCCCValence.result()
        print("Training arousal CCC over epoch: %.4f - Valence: %.4f" % (float(train_cccArousal),float(train_cccValence)))
        model.save(modelFolder)

        # Reset training metrics at the end of each epoch
        trainCCCArousal.reset_states()
        trainCCCValence.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in validationGenerator:
            modelArousal = Model(inputs=model.input, outputs=model.get_layer("arousal_output").output)
            test_step(x_batch_val, y_batch_val, modelArousal, testCCCArousal)

            modelValence = Model(inputs=model.input, outputs=model.get_layer("valence_output").output)
            test_step(x_batch_val, y_batch_val, modelArousal, testCCCValence)


        val_cccArousal = testCCCArousal.result()
        val_cccValence = testCCCValence.result()

        val_cccArousal.reset_states()
        val_cccValence.reset_states()
        print("Validation CCC Arousal: %.4f - Valence: %.4f" % (float(val_cccArousal),float(val_cccValence)))
        print("Time taken: %.2fs" % (time.time() - start_time))



    return model

def trainCategorical(model, saveFolder, trainGenerator, validationGenerator,  trainingSetup, trainWithCIAO, config):


    # Adjust the classifier updatable parameters
    model = setTrainingLayerSetup(model, trainingSetup, config["lastConvName"],config["denseLayers"] )

    print("----------------")
    print("Training classifier model:")
    model.summary()
    print("----------------")


    if trainWithCIAO:
        # Adjust the CIAO updatable parameters
        model = setCIAOTrainingSetup(model)

        print("----------------")
        print("Training CIAO model:")
        model.summary()
        print("----------------")

    """Create folders for training"""
    modelFolder = saveFolder+"/Model"
    createFolders(modelFolder)


    """Original model train parameters"""
    # Instantiate the optmizer
    if config["optmizer"] == "SGD":
      optmizer = SGD(config["initialLearningRate"], momentum=config["momentum"], nesterov=config["nesterov"])
    elif config["optmizer"] == "Adam":
        optmizer = Adam(config["initialLearningRate"])

    # Instantiate lossFunctions
    lossCat =  tf.keras.losses.CategoricalCrossentropy(from_logits=True)


    """CIAO model train parameters
    """
    if trainWithCIAO:
        # Instantiate the optmizer
        if config["optmizer"] == "SGD":
            optmizerCIAO = SGD(config["initialLearningRate"], momentum=config["momentum"], nesterov=config["nesterov"])
        elif config["optmizer"] == "Adam":
            optmizerCIAO = Adam(config["initialLearningRate"])

        # Instantiate lossFunctions
        lossCIAO = SupervisedContrastiveLoss(config["temperature"], SupervisedContrastiveLoss.CLASSIFIERTYPE["categorical"])


    #Prepare the Metrics
    trainAccuracy = tf.keras.metrics.CategoricalAccuracy()
    testAccuracy = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(config["epoches"]):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(trainGenerator):

            totalLoss = 0
            # If train with CIAO, first
            if trainWithCIAO:
                model = setCIAOTrainingSetup(model)
                loss_value_CIAO = train_step(x_batch_train, y_batch_train, model, optmizerCIAO, lossCIAO, trainAccuracy)

                model = setTrainingLayerSetup(model, trainingSetup, config["lastConvName"],config["denseLayers"])

            """Train the original Model"""
            loss_value = train_step(x_batch_train, y_batch_train, model, optmizer, lossCat, trainAccuracy)


            # Log every 200 batches.
            if step % 200 == 0:
                if trainWithCIAO:
                    print(
                        "Training loss (for one batch) at step %d: %.4f - CIAO loss:%.4f "
                        % (step, float(loss_value), float(loss_value_CIAO))
                    )
                else:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                print("Seen so far: %d samples" % ((step + 1) * config["batchSize"]))

        # Display metrics at the end of each epoch.
        train_acc = trainAccuracy.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        model.save(modelFolder)

        # Reset training metrics at the end of each epoch
        trainAccuracy.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in validationGenerator:
            test_step(x_batch_val, y_batch_val, model, testAccuracy)

        val_acc = testAccuracy.result()
        testAccuracy.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))



    return model


def predict(model, testGenerator):

    predictions = model.predict(testGenerator)

    return predictions


def evaluate(model, validationGenerator, batchSize):

    scores = model.evaluate(validationGenerator, batch_size=batchSize)
    return scores

"""Categorical"""
def getCategoricalClassifier(encoder, withCIAO, config):

        #Add CIAO
        if withCIAO:
            last_layer = addCIAO(encoder,  config)
        else:
            last_layer = encoder.output

        x = MaxPooling2D(pool_size=(2, 2))(last_layer)
        x = Flatten(name='flatten')(x)

        for i in range(config["denseLayers"]):
            x = Dense(config["denseLayerSize"], activation="relu", name="denseLayer" + str(i))(x)
            x = Dropout(0.5)(x)

        categoricalOutput = Dense(units=config["outputSize"], activation="softmax", name='category_output')(x)

        # Build models and adjust the trainable parameters of the encoders
        model = Model(inputs=encoder.input, outputs=categoricalOutput)

        if withCIAO:
            model.get_layer('Inhibition_CIAO').set_weights(model.get_layer(
                config["lastConvName"]).get_weights())  # Guarantees that the inbitorylayer and the convlayer starts with the same weights!

        return model



"""A/V"""
def getAVClassifier(encoder, withCIAO, config):

        #Add CIAO
        if withCIAO:
            last_layer = addCIAO(encoder,  config)
        else:
            last_layer = encoder.output

        x = MaxPooling2D(pool_size=(2, 2))(last_layer)
        x = Flatten(name='flatten')(x)

        for i in range(config["denseLayers"]):
            x = Dense(config["denseLayerSize"], activation="relu", name="denseLayer" + str(i))(x)
            x = Dropout(0.5)(x)

        denseA = Dense(config["denseLayerSize"], activation="relu", name="denseLayer_A")(x)
        drop6 = Dropout(0.5)(denseA)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop6)

        denseV = Dense(config["denseLayerSize"], activation="relu", name="denseLayer_V")(x)
        drop7 = Dropout(0.5)(denseV)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(drop7)


        # Build models and adjust the trainable parameters of the encoders
        model = Model(inputs=encoder.input, outputs=[arousal_output, valence_output])

        if withCIAO:
            model.get_layer('Inhibition_CIAO').set_weights(model.get_layer(
                config["lastConvName"]).get_weights())  # Guarantees that the inbitorylayer and the convlayer starts with the same weights!

        return model



"""Binary CrossEntropy Sequential"""
def getBinaryCrossEntropySequential(encoder, withCIAO, config):

        #Add CIAO
        if withCIAO:
            last_layer = addCIAO(encoder,  config)
        else:
            last_layer = encoder.output

        x = MaxPooling2D(pool_size=(2, 2))(last_layer)
        x = Flatten(name='flatten')(x)

        input_layer = Input(shape=[config["sequenceSize"],config["imgSize"][0], config["imgSize"][1], config["imgSize"][2]])

        td1 = TimeDistributed(x, name="encoder")(input_layer)

        # Flatten
        flatten = TimeDistributed(Flatten(), name="Flatten")(td1)

        # RNN

        x = LSTM(config["LSTMLayerSize"], activation='relu', return_sequences=False, name="Rnn_1")(flatten)

        for i in range(config["denseLayers"]):
            x = Dense(config["denseLayerSize"], activation="relu", name="denseLayer" + str(i))(x)
            x = Dropout(0.5)(x)

        outputs = []

        for a in range(config["outputSize"]):
            output1 = Dense(units=1, activation="softmax", name="class" + str(a) + "_output")(x)
            outputs.append(output1)

        model = Model(inputs=input_layer, outputs=outputs)

        if withCIAO:
            model.get_layer('Inhibition_CIAO').set_weights(model.get_layer(
                config["lastConvName"]).get_weights())  # Guarantees that the inbitorylayer and the convlayer starts with the same weights!

        return model

