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

IMAGE_METADATA_CSV = 'data/driving_log.csv'
IMAGE_PATH = 'data/IMG/'
FOLDER_SEPARATOR = '\\'
TURNING_OFFSET = 0.25
TRAIN_VALID_SPLIT = 0.2
TRAIN_EPOCHS = 1
LEARN_RATE = 0.001
BATCH_SIZE = 512
USE_GENERATOR = False


def generator(samples):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, BATCH_SIZE):
            batch_samples = samples[offset:offset + BATCH_SIZE]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    images = []
                    measurements = []
                    source_path = batch_sample[i]
                    filename = source_path.split(FOLDER_SEPARATOR)[-1]
                    current_path = IMAGE_PATH + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3])
                    if i == 0:
                        measurements.append(measurement)
                    elif i == 1:
                        measurements.append(measurement - TURNING_OFFSET)
                    elif i == 2:
                        measurements.append(measurement + TURNING_OFFSET)

            # create more test data by flipping the images and inverting the corresponding turn angles
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                # only add a flipped turning image
                if (measurement < -0.1 or measurement > 0.1):
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(measurement * -1.0)

            X_train_gen = np.array(augmented_images)
            y_train_gen = np.array(augmented_measurements)
            yield shuffle(X_train_gen, y_train_gen)


# import measurements csv
lines = []
with open(IMAGE_METADATA_CSV) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

if USE_GENERATOR:
    # use generators
    train_samples, validation_samples = train_test_split(lines, test_size=TRAIN_VALID_SPLIT)
    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

else:
    # create arrays of images and measurements
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split(FOLDER_SEPARATOR)[-1]
            current_path = IMAGE_PATH + filename
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            if i == 0:
                measurements.append(measurement)
            elif i == 1:
                measurements.append(measurement - TURNING_OFFSET)
            elif i == 2:
                measurements.append(measurement + TURNING_OFFSET)

    # create more test data by flipping the images and inverting the corresponding turn angles
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        # only add a flipped turning image
        # if (measurement.value() < -0.1 or measurement > 0.1):
        #augmented_images.append(cv2.flip(image, 1))
        #augmented_measurements.append(measurement * -1.0)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

model = Sequential()
# TODO: Figure out, if Cropping before Lambda is good
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(75, 320, 3)))

model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))

#model.add(Dropout(0.5))

model.add(Flatten())

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
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=TRAIN_EPOCHS)
else:
    # train model
    history_object = model.fit(X_train, y_train, validation_split=TRAIN_VALID_SPLIT, shuffle=True, epochs=TRAIN_EPOCHS, verbose=1)
    # print layer information
    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

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
