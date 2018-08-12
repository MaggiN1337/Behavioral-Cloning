import sys
import csv
import cv2
import numpy as np
from collections import Counter
from PIL import Image, ImageEnhance
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, Activation
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# file system settings
IMG_PATH = 'track1/IMG/'
IMAGE_METADATA_CSV = 'track1/driving_log.csv'
TRACK2_METADATA_CSV = 'track2/driving_log.csv'
FOLDER_SEPARATOR = '\\'
OUTPUT_PATH = 'examples/'

# network parameters
TURNING_OFFSET = 0.25
LIMIT_IMAGES_PER_TURNING_ANGLE = 400
TRAIN_VALID_SPLIT = 0.2
TRAIN_EPOCHS = 5
LEARN_RATE = 0.001

# train with generators
BATCH_SIZE = 512
USE_GENERATOR = False  # type: bool

# training image editing
FLIP_IMAGES = True  # type: bool
ADAPT_BRIGHTNESS = True  # type: bool
USE_GAMMA_CORRECTION = False  # type: bool
USE_TRACK2 = True  # type: bool

# debug settings
DEBUG = True  # type: bool
LIMIT_IMAGES_FOR_DEBUGGING = 40000

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
            lines.append(line)
    return lines


# method for example image output
def save_images_as_png(array_with_images):
    for i in range(len(array_with_images)):
        cv2.imwrite(OUTPUT_PATH + 'image_' + str(i) + '.png', array_with_images[i])


# method for image augmentation
def image_augmentation(images_as_array, augment_data):
    augmented_images, augmented_measurements = [], []
    counter = 0
    for line in images_as_array:
        if DEBUG and (len(augmented_images) > LIMIT_IMAGES_FOR_DEBUGGING):
            print("Image limitation for debugging is reached.")
            break

        # calculate training data distribution
        distribution = Counter(augmented_measurements)

        # as each line contains 3 images, get each of them with its turning angle
        for i in range(3):
            image_from_line, meas_from_line = get_image_and_measurement_from_line(line, i)

            if distribution[meas_from_line] >= LIMIT_IMAGES_PER_TURNING_ANGLE:
                # print("Already enough images for turning angle %f" % meas_from_line)
                break

            augmented_images.append(image_from_line)
            augmented_measurements.append(meas_from_line)

            # create more test data by flipping the images and inverting the corresponding turn angles
            # do not augment data, if already enough
            if augment_data:

                # only add a flipped image of a turning image
                if FLIP_IMAGES and abs(meas_from_line) > 0.2:
                    augmented_images.append(cv2.flip(image_from_line, 1))
                    augmented_measurements.append(meas_from_line * -1.0)

                # add a bright and dark image to the data set
                if USE_GAMMA_CORRECTION:
                    augmented_images.append(gamma_correction(image_from_line, -5))
                    augmented_measurements.append(meas_from_line)

                    augmented_images.append(gamma_correction(image_from_line, 5))
                    augmented_measurements.append(meas_from_line)

                if ADAPT_BRIGHTNESS:
                    # darken images
                    augmented_images.append(adjust_brightness(image_from_line, 1.6))
                    augmented_measurements.append(meas_from_line)

                    # brighten images
                    augmented_images.append(adjust_brightness(image_from_line, 0.4))
                    augmented_measurements.append(meas_from_line)

        # save only images of first run
        if counter == 0:
            save_images_as_png(augmented_images)
            counter += 1
        # count number of processed images (3 for each line) and print every 500
        counter += 3
        if counter % 500 == 0:
            print("Processed %d images." % counter)
    return augmented_images, augmented_measurements


# load image from filename and get corresponding turning angle
def get_image_and_measurement_from_line(line, i):
    source_path = line[i]
    filename = source_path.split(FOLDER_SEPARATOR)[-1]
    relative_path_to_image = IMG_PATH + filename

    # load image
    image = cv2.imread(relative_path_to_image)
    # cv2.imwrite(OUTPUT_PATH + 'input_image.png', image)

    # TODO: crop images here

    # convert to RGB color for drive.py
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(OUTPUT_PATH + 'input_image_RGB.png', image)

    if USE_GENERATOR:
        # crop image here
        image = image[60:140, 0:320]

    # center image
    meas = float(line[3])
    # left image
    if i == 1:
        meas += TURNING_OFFSET
    # right image
    elif i == 2:
        meas -= TURNING_OFFSET
    return image, meas


# adjust image brightness to simulate sun and shadow
def adjust_brightness(input_image, factor):
    image = Image.fromarray(np.uint8(input_image))
    enhancer_object = ImageEnhance.Brightness(image)
    out = enhancer_object.enhance(factor)
    return np.array(out)


# apply gamma correction for shadow simulation
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
def generator(samples, batch_size=BATCH_SIZE, augment_data=False):
    X_train_gen, y_train_gen = [], []
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    i = -1
    while 1:
        X_train_gen, y_train_gen = [], []
        shuffle(samples)
        i += 1
        y = 0
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            # print("\nNext Batch run %d / %d " % (offset, num_samples))
            # print("\nNext Batch run %d.%d " % (i, y))

            X_train_temp, y_train_temp = image_augmentation(batch_samples, augment_data)

            X_train_gen = np.array(X_train_temp)
            y_train_gen = np.array(y_train_temp)
        # TODO: fix yield one level higher
        yield shuffle(X_train_gen, y_train_gen)
        y += 1


# define keras model
def create_model():
    global model
    model = Sequential()
    if USE_GENERATOR:
        model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(80, 320, 3)))
    else:
        # crop 60 pixels from top, 20 from bottom, 0 from left and right
        model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80, 320, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", activation="relu"))

    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1000))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation("relu"))

    model.add(Dense(10))
    model.add(Activation("relu"))

    model.add(Dense(1))

    # use mean squared error function with adam-optimizer
    model.compile(loss='mse', optimizer=Adam(LEARN_RATE))

    return model


# plot a distribution from a list of key-value-pairs with a specific title
def plot_counts_as_png(measurements, title):
    data, counts = zip(*measurements)
    plt.bar(data, counts, 0.05, align='center')
    plt.title(title)
    plt.savefig('training_data_distribution.png')
    plt.close()


# save the training and validation loss as image
def plot_loss_functions_as_png(training):
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, TRAIN_EPOCHS, step=1.0))
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('loss_visualization.png')
    plt.close()


# write input variables and model summary to logfile
def export_execution_details():
    with open(LOGFILE_NAME, 'w') as f:
        with redirect_stdout(f):
            print("This run had the following settings:")
            print("Track2 was used: " + str(USE_TRACK2))
            print("Image augmentation was used with: ")
            print(" - flipping images: " + str(FLIP_IMAGES))
            print(" - gamma correction: " + str(USE_GAMMA_CORRECTION))
            print(" - Brightness adjustment: " + str(ADAPT_BRIGHTNESS))
            print("Number of epochs: " + str(TRAIN_EPOCHS))
            print("Learning rate: " + str(LEARN_RATE))
            print("Turning angle offset: " + str(TURNING_OFFSET))
            print("Image limitation per turning angle: " + str(LIMIT_IMAGES_PER_TURNING_ANGLE))
            print("Train validation split ratio: " + str(TRAIN_VALID_SPLIT))
            print("Generator was used: " + str(USE_GENERATOR))
            if USE_GENERATOR:
                print("Batch size: " + str(BATCH_SIZE))
            print("Keras layer summary: ")
            model.summary()


# import track data
track_data = csv_to_array(IMAGE_METADATA_CSV)
if USE_TRACK2:
    track_data.extend(csv_to_array(TRACK2_METADATA_CSV))

# shuffle all the track data
np.random.shuffle(track_data)

print("Listed %d raw images now." % (3 * len(track_data)))

# define generators or preprocess images
if USE_GENERATOR:
    # use generators
    train_samples, validation_samples = train_test_split(track_data, test_size=TRAIN_VALID_SPLIT)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE, augment_data=True)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, augment_data=False)
else:
    # create arrays of images and measurements
    X_train, y_train = image_augmentation(track_data, augment_data=True)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # print test data distribution
    plot_counts_as_png(Counter(y_train).items(), "Turning angle measurement distribution")
    print("Train the network with %d images now." % len(y_train))

# clean up raw track data
track_data = []

# create the learning model
create_model()

# Train the model with or without generator
if USE_GENERATOR:
    history_object = create_model().fit_generator(train_generator,
                                                  steps_per_epoch=len(train_samples),
                                                  validation_data=validation_generator,
                                                  validation_steps=len(validation_samples),
                                                  epochs=TRAIN_EPOCHS,
                                                  verbose=1,
                                                  # callbacks=[csv_logger],
                                                  # max_queue_size=3,
                                                  # workers=4
                                                  # use_multiprocessing=False
                                                  )
else:
    history_object = create_model().fit(X_train,
                                        y_train,
                                        validation_split=TRAIN_VALID_SPLIT,
                                        shuffle=True,
                                        epochs=TRAIN_EPOCHS,
                                        verbose=1,
                                        # callbacks=[csv_logger]
                                        )

# print layer information
export_execution_details()

# save trained model to file
model.save('model.h5')

plot_loss_functions_as_png(history_object)
