import setGPU

import tensorflow as tf

import matplotlib.pyplot as plt
from random import shuffle
import os
import numpy as np
import cv2 as cv

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../data/sci_tvt/training", "pass the training data path")
flags.DEFINE_string("validation_data_path", "../data/sci_tvt/validation", "pass the valitaion data path")
flags.DEFINE_string("model_path", "../model/cnn_class_model.h5", "path where model needs to be stored")

class train_network(object):
    def __init__(self):
        self.classes = 3
        self.learning_rate = 0.0001
        self.EPOCHS = 500
        self.BATCH_SIZE = 64
        self.IMG_WIDTH = self.IMG_HEIGHT = 192
        self.CHANNELS = 3

    def inference(inp_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=inp_shape, name='conv1'))
        model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation = 'relu', name='conv2'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv3'))
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation = 'relu', name='conv4'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv5'))
        model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation = 'relu', name='conv6'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',  name='conv7'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation = 'relu', name='conv8'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation = 'relu', name='conv9'))
        model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation = 'relu', name='conv10'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(256, activation='relu', name='dense2'))
        model.add(tf.keras.layers.Dense(self.classes, activation='softmax', name='final_class_dense'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss= 'categorical_crossentropy', metrics=['acc'] )

        return model


    def train():
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                                               horizontal_flip=True, fill_mode='nearest')
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(FLAGS.train_data_path, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT), batch_size=self.BATCH_SIZE, class_mode='categorical')
        valid_generator = valid_datagen.flow_from_directory(FLAGS.validation_data_path, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT), batch_size=self.BATCH_SIZE, class_mode='categorical')

        input_shape = (self.IMG_WIDTH, self.IMG_HEIGHT, self.CHANNELS)
        model = inference(input_shape)
        print(model.summary())

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, min_lr=0.001)
        callbacks = [es_callback,  reduce_lr]

        train_samples_batch = train_generator.n//self.BATCH_SIZE
        valid_samples_batch = valid_generator.n//self.BATCH_SIZE

        history = model.fit_generator(train_generator, steps_per_epoch=train_samples_batch, epochs=self.EPOCHS, validation_data=valid_generator,
                              validation_steps=valid_samples_batch, callbacks=callbacks)

        model.save(FLAGS.model_path)

