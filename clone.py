import csv
import cv2
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# save whole output to file
#sys.stdout = open('modelsummary.txt', 'w')

IMG_PATH = 'track1/IMG/'
IMAGE_METADATA_CSV = 'track1/driving_log.csv'
TRACK2_METADATA_CSV = 'track2/driving_log.csv'

FOLDER_SEPARATOR = '\\'
TURNING_OFFSET = 0.25
TRAIN_VALID_SPLIT = 0.2
TRAIN_EPOCHS = 3
LEARN_RATE = 0.001
BATCH_SIZE = 32
FLIP_IMAGES = True
USE_GENERATOR = False
USE_TRACK2 = False


# method to import and measurements from csv
def csv_to_array(filename):
    lines = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
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
        if FLIP_IMAGES and (measurement < -0.1 or measurement > 0.1):
            augmented_images.append(cv2.flip(image, 1))
            augmented_measurements.append(measurement * -1.0)
    return augmented_images, augmented_measurements


# TODO: add gamma correction for shadow simulation


# method for batch processing with generators
def generator(samples):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, BATCH_SIZE):
            batch_samples = samples[offset:offset + BATCH_SIZE]

            X_train_temp, y_train_temp = image_augmentation(batch_samples)

            X_train_gen = np.array(X_train_temp)
            y_train_gen = np.array(y_train_temp)
            yield shuffle(X_train_gen, y_train_gen)


track1_data = csv_to_array(IMAGE_METADATA_CSV)
if USE_TRACK2:
    track2_data = csv_to_array(TRACK2_METADATA_CSV)
    track1_data.extend(track2_data)

if USE_GENERATOR:
    # use generators
    train_samples, validation_samples = train_test_split(track1_data, test_size=TRAIN_VALID_SPLIT)
    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

else:
    # create arrays of images and measurements
    X_train, y_train = image_augmentation(track1_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# crop 60 pixels from top, 20 from bottom, 0 from left and right
model.add(Cropping2D(cropping=((60, 20), (0, 0))))

model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))

# model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1000))
model.add(Activation("relu"))
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
                                         verbose=1)
else:
    # train model
    history_object = model.fit(X_train,
                               y_train,
                               validation_split=TRAIN_VALID_SPLIT,
                               shuffle=True,
                               epochs=TRAIN_EPOCHS,
                               verbose=1)

# print layer information
with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        print("Number of epochs: " + str(TRAIN_EPOCHS))
        print("Learning rate: " + str(LEARN_RATE))
        print("Batch size: " + str(BATCH_SIZE))

# save trained model to file
model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# save the training and validation loss as image
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_visualization.png')
plt.close()
