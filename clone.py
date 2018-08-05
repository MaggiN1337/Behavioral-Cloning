import csv
import sys
from contextlib import redirect_stdout

import cv2
import numpy as np
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, Activation
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# file system settings
IMG_PATH = 'track1/IMG/'
IMAGE_METADATA_CSV = 'track1/driving_log.csv'
TRACK2_METADATA_CSV = 'track2/driving_log.csv'
FOLDER_SEPARATOR = '\\'

# network parameters
TURNING_OFFSET = 0.25
TRAIN_VALID_SPLIT = 0.2
TRAIN_EPOCHS = 3
LEARN_RATE = 0.001
BATCH_SIZE = 32

# learning settings
FLIP_IMAGES = True
USE_GAMMA_CORRECTION = True
USE_TRACK2 = False
USE_GENERATOR = False

# debug settings
DEBUG = True
LIMIT_IMAGES_FOR_DEBUGGING = 128

# settings for logging
LOGFILE_NAME = 'logfile.txt'
csv_logger = CSVLogger('log.csv', append=False, separator=';')


# save whole stdout to file
# sys.stdout = open(LOGFILE_NAME, 'w')


# method to import and measurements from csv
def csv_to_array(filename):
    lines = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if DEBUG and (len(lines) == LIMIT_IMAGES_FOR_DEBUGGING):
                break
            lines.append(line)
    return lines


# method for image augmentation
def image_augmentation(images_as_array):
    images = []
    measurements = []
    for line in images_as_array:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split(FOLDER_SEPARATOR)[-1]
            current_path = IMG_PATH + filename
            image = cv2.imread(current_path)

            if USE_GENERATOR:
                # crop image here
                image = image[60:140, 0:320]

            images.append(image)
            measurement = float(line[3])
            # center image
            if i == 0:
                measurements.append(measurement)
            # left image
            elif i == 1:
                measurements.append(measurement + TURNING_OFFSET)
            # right image
            elif i == 2:
                measurements.append(measurement - TURNING_OFFSET)
    # create more test data by flipping the images and inverting the corresponding turn angles
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        # only add a flipped image of a turning image
        if (FLIP_IMAGES and abs(measurement) > 0.2):
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement * -1.0)
        if USE_GAMMA_CORRECTION:
            # darken images
            for y in range(-5, 0, 2):
                augmented_images.append(gamma_correction(image, y))
                augmented_measurements.append(measurement)
            # brigthen images
            for y in range(1, 6, 2):
                augmented_images.append(gamma_correction(image, y))
                augmented_measurements.append(measurement)
    return augmented_images, augmented_measurements


# TODO: add gamma correction for shadow simulation
def gamma_correction(img, gamma):
    # brighten or darken image
    inv_gamma = 1.0 / gamma

    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255
         for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


# method for batch processing with generators
# TODO: Fix generator for long duration
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            print("Next Batch run with: ", offset)

            X_train_temp, y_train_temp = image_augmentation(batch_samples)

            X_train_gen = np.array(X_train_temp)
            y_train_gen = np.array(y_train_temp)
            yield shuffle(X_train_gen, y_train_gen)


track_data = csv_to_array(IMAGE_METADATA_CSV)
if USE_TRACK2:
    track2_data = csv_to_array(TRACK2_METADATA_CSV)
    track_data.extend(track2_data)
    track2_data = []

if USE_GENERATOR:
    # use generators
    train_samples, validation_samples = train_test_split(track_data, test_size=TRAIN_VALID_SPLIT)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

else:
    # create arrays of images and measurements
    X_train, y_train = image_augmentation(track_data)
    track_data = []

    X_train = np.array(X_train)
    y_train = np.array(y_train)

model = Sequential()
if USE_GENERATOR:
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(80, 320, 3)))
    # , output_shape=(80, 320, 3)
else:
    model.add(Cropping2D(cropping=((60, 20), (0, 0))))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80, 320, 3)))
    # crop 60 pixels from top, 20 from bottom, 0 from left and right

model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(Conv2D(64, (3, 3), activation="relu"))

# model.add(Dropout(0.5))

model.add(Flatten())

# model.add(Dense(1000))
# model.add(Activation("relu"))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))

# use mean squared error function with adam-optimizer
model.compile(loss='mse', optimizer=Adam(LEARN_RATE))

if USE_GENERATOR:
    # use generator to train model
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         validation_steps=len(validation_samples),
                                         epochs=TRAIN_EPOCHS,
                                         verbose=1,
                                         callbacks=[csv_logger])
else:
    # train model
    history_object = model.fit(X_train,
                               y_train,
                               validation_split=TRAIN_VALID_SPLIT,
                               shuffle=True,
                               epochs=TRAIN_EPOCHS,
                               verbose=1,
                               callbacks=[csv_logger])

# print layer information
with open(LOGFILE_NAME, 'w') as f:
    with redirect_stdout(f):
        print("This run had the following settings:")
        print("Generator was used: " + str(USE_GENERATOR))
        print("Track2 was used: " + str(USE_TRACK2))
        print("Image augmentation was used with flipping images: " + str(FLIP_IMAGES))
        print("Number of epochs: " + str(TRAIN_EPOCHS))
        print("Learning rate: " + str(LEARN_RATE))
        print("Batch size: " + str(BATCH_SIZE))

        print("Keras layer summary: ")
        model.summary()

# save trained model to file
model.save('model.h5')

# save the training and validation loss as image
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_visualization.png')
plt.close()
