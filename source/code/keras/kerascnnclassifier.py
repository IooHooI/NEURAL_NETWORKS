import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from .kerasbaseestimator import KerasBaseEstimator


class KerasCNNClassifier(KerasBaseEstimator):

    def __init__(self, checkpoint_dir='./', lr=0.01, batch_size=128, n_epochs=12):
        super().__init__(checkpoint_dir, lr, batch_size, n_epochs)

    def build_the_graph(self, input_shape, output_shape):
        self.model.add(Conv2D(128, (3, 3), input_shape=input_shape, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # self.model.add(Conv2D(256, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_shape, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit(self, X, y=None):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.3
        )
        train_generator = train_datagen.flow_from_directory(
            X,
            target_size=(200, 200),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )

        self.build_the_graph(train_generator.image_shape, 1)

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n / self.batch_size,
            epochs=self.n_epochs,
            callbacks=[
                ModelCheckpoint(
                    filepath=self.checkpoint_dir,
                    save_best_only=True,
                    verbose=1,
                    monitor='acc'
                ),
                EarlyStopping(
                    patience=3,
                    verbose=1,
                    monitor='acc'
                ),
                ReduceLROnPlateau(
                    min_lr=0.0001,
                    mode='auto',
                    verbose=1,
                    monitor='acc'
                )
            ]
        )

    def predict(self, X, y=None):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            X,
            target_size=(200, 200),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        y_proba = self.model.predict_generator(test_generator, verbose=1)
        return test_generator.classes, np.array([int(round(proba[0])) for proba in y_proba])

    def predict_proba(self, X, y=None):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            X,
            target_size=(200, 200),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        y_proba = self.model.predict_generator(test_generator, verbose=1)
        return test_generator.classes, np.array([proba[0] for proba in y_proba])
