import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

import gen_data


def run(d):
    IMG_WIDTH = d
    IMG_HEIGHT = d
    IMG_CHANNELS = 1
    mu_1 = 0
    mu_2 = 1

    X_train, Y_train = gen_data.generate(5000, IMG_WIDTH, mu_1, mu_2)

    print(X_train.shape, Y_train.shape)

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(4, (3, 3), padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    u2 = UpSampling2D(size=(2, 2))(p1)
    c2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u2)

    outputs = c2

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    earlystopper = EarlyStopping(patience=15, verbose=1)
    checkpointer = ModelCheckpoint('./model/test_' + str(d) + '.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, epochs=20,
                        callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    list_d = [4, 8, 16, 32]

    for d in list_d:
        run(d)