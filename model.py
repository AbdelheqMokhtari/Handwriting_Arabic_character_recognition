import os
import cv2 as cv
import numpy as np
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense,\
    Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator


# Loading Train Data
img_train_path = r'Train'
train_data = []
train_label_names = []
for directory in os.listdir(img_train_path):
    for file in os.listdir(os.path.join(img_train_path, directory)):
        image_path = os.path.join(img_train_path, directory, file)
        image = cv.imread(image_path, 0)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        train_data.append(image)
        train_label_names.append(directory)


target_dict = {k: v for v, k in enumerate(np.unique(train_label_names))}
train_label = [target_dict[train_label_names[i]] for i in range(len(train_label_names))]

row = len(train_data)
height = np.shape(train_data)
width = len(train_data[1])
print(f'Images:{row}, height:{height}, width:{width}')
print("Shape of a list:", len(train_data))


# Loading Test Data
img_test_path = r'Test'
test_data = []
test_label_names = []
for directory in os.listdir(img_test_path):
    for file in os.listdir(os.path.join(img_test_path, directory)):
        image_path = os.path.join(img_test_path, directory, file)
        image = cv.imread(image_path, 0)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        test_data.append(image)
        test_label_names.append(directory)


target_dict = {k: v for v, k in enumerate(np.unique(test_label_names))}
test_label = [target_dict[test_label_names[i]] for i in range(len(test_label_names))]
row = len(test_data)
height = len(test_data[0])
width = len(test_data[1])
print(f'Images:{row}, height:{height}, width:{width}')
print("Shape of a list:", len(test_data))


train_label = to_categorical(train_label)
test_label = to_categorical(test_label)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation ='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5)))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5)))
model.add(Dropout(0.3))
model.add(Dense(28, activation="softmax"))

optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
batch_size = 68

# Fit the model
# history = model.fit_generator(train_data, train_label, batch_size=batch_size,
#                              epochs=epochs, validation_data=(test_data, test_label),
#                              steps_per_epoch=test_data.shape[0] // batch_size,
#                              verbose=2, )

