import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

IMAGE_METADATA_CSV = 'driving_log.csv'
IMAGE_PATH = 'IMG/'
TURNING_OFFSET = 0.2

# import measurements csv
lines = []
with open(IMAGE_METADATA_CSV) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

# create arrays of images and measurements
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
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
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
# TODO: Figure out, if Cropping before Lambda is good
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(75, 320, 3)))
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# loss function
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=5)

model.save('model.h5')
