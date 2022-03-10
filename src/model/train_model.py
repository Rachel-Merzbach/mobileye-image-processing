""" define and train the Model and save it as model.h5 file """
# given jupyter file #

import numpy as np, matplotlib.pyplot as plt
from os.path import join
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, BatchNormalization, Activation, Conv2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


# define the model
def tfl_model():
    input_shape = (81, 81, 3)
    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for _ in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 64)
    spatial_layer(2, 64)
    spatial_layer(2, 96)
    spatial_layer(1, 96)

    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model


m = tfl_model()
m.summary()

# train the model
DATA_DIR = r'../data'
datasets = {
    'val': load_tfl_data(join(DATA_DIR, 'val')),
    'train': load_tfl_data(join(DATA_DIR, 'train')),
}

m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

train, val = datasets['train'], datasets['val']
m.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=5)

# save the training model
m.save("model.h5")
