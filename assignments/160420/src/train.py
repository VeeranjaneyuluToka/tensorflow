import setGPU

import tensorflow as tf

import matplotlib.pyplot as plt
from random import shuffle
import os
import numpy as np
import cv2 as cv

from feature_eng import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_path", "../data/sci_tvt/training", "pass the training data path")
flags.DEFINE_string("validation_data_path", "../data/sci_tvt/validation", "pass the valitaion data path")
flags.DEFINE_string("model_path", "../artifacts/cnn_class_model.h5", "path where model needs to be stored")

class train_network(object):
    def __init__(self):
        self.classes = 3
        self.learning_rate = 0.0001
        self.EPOCHS = 500
        self.BATCH_SIZE = 64
        self.IMG_WIDTH = self.IMG_HEIGHT = 192
        self.CHANNELS = 1

    def inference(self, inp_shape):
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


    def train(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                                               horizontal_flip=True, fill_mode='nearest')
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(FLAGS.train_data_path, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT), batch_size=self.BATCH_SIZE,
                color_mode='grayscale', class_mode='categorical')
        valid_generator = valid_datagen.flow_from_directory(FLAGS.validation_data_path, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT), batch_size=self.BATCH_SIZE, 
                color_mode='grayscale', class_mode='categorical')

        input_shape = (self.IMG_WIDTH, self.IMG_HEIGHT, self.CHANNELS)
        model = self.inference(input_shape)
        print(model.summary())

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, min_lr=0.001)
        callbacks = [es_callback,  reduce_lr]

        train_samples_batch = train_generator.n//self.BATCH_SIZE
        valid_samples_batch = valid_generator.n//self.BATCH_SIZE

        history = model.fit_generator(train_generator, steps_per_epoch=train_samples_batch, epochs=self.EPOCHS, validation_data=valid_generator,
                              validation_steps=valid_samples_batch, callbacks=callbacks)

        model.save(FLAGS.model_path)

class data_pipeline(object):
    def __init__(self):
        self.fv_path = ""
        self.nv_path = ""
        self.pv_path = ""

        """ paths where augmented files stored and augmented operations run through """
        self.fv_aug_data_path = '../data/aug_data/FULL_VISIBILITY'
        self.nv_aug_data_path = '../data/aug_data/NO_VISIBILITY'
        self.pv_aug_data_path = '../data/aug_data/PARTIAL_VISIBILITY'

        self.roperation = 'rotation'
        self.boperation = 'brightness'
        self.zoperation = 'zoom'

    def merge_original_aug_data(self, da_obj):
        """ merge full visibility class data"""
        fv_aug_data_path_r = os.path.join(self.fv_aug_data_path, self.roperation) #rotated data
        da_obj.merge_original_rotated_data(fv_aug_data_path_r, self.fv_path)

        fv_aug_data_path_b = os.path.join(self.fv_aug_data_path, self.boperation) #brightness data
        da_obj.merge_original_rotated_data(fv_aug_data_path_b, self.fv_path)

        fv_aug_data_path_z = os.path.join(self.fv_aug_data_path, self.zoperation) #zoom data
        da_obj.merge_original_rotated_data(fv_aug_data_path_z, self.fv_path)

        """ merge no-visibility class data, skipping rotoation as already merged while augmenting in case no and partial visibility """
        nv_aug_data_path_b = os.path.join(self.nv_aug_data_path, self.boperation) #brightness data
        da_obj.merge_original_rotated_data(nv_aug_data_path_b, self.nv_path)

        nv_aug_data_path_z = os.path.join(self.nv_aug_data_path, self.zoperation) #zoom data
        da_obj.merge_original_rotated_data(nv_aug_data_path_z, self.nv_path)

        """ merge partial-visibility class data"""
        pv_aug_data_path_b = os.path.join(self.pv_aug_data_path, self.boperation) #brightness data
        da_obj.merge_original_rotated_data(pv_aug_data_path_b, self.pv_path)

        pv_aug_data_path_z = os.path.join(self.pv_aug_data_path, self.zoperation) #zoom data
        da_obj.merge_original_rotated_data(pv_aug_data_path_z, self.pv_path)


    def data_augment_merge(self, da_obj):
        """ FULL visibaility data augmentation """
        class_type = 'full_vis'

        fv_angles = [90]
        da_obj.trigger_augmentation(self.fv_aug_data_path, self.fv_path, fv_angles, class_type, self.roperation) #rotation
        fv_angles = [0]
        da_obj.trigger_augmentation(self.fv_aug_data_path, self.fv_path, fv_angles, class_type, self.boperation) #brightness
        da_obj.trigger_augmentation(self.fv_aug_data_path, self.fv_path, fv_angles, class_type, self.zoperation) #zoom

        """ NO visibility data augmentation """
        class_type = 'no_vis'

        nv_angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
        da_obj.trigger_augmentation(self.nv_aug_data_path, self.nv_path, nv_angles, class_type, self.roperation) #rotation
        nv_aug_data_path_n = os.path.join(self.nv_aug_data_path, self.roperation)
        da_obj.merge_original_rotated_data(nv_aug_data_path_n, self.nv_path)

        nv_angles = [0]
        da_obj.trigger_augmentation(self.nv_aug_data_path, self.nv_path, nv_angles, class_type, self.boperation) #brightness
        da_obj.trigger_augmentation(self.nv_aug_data_path, self.nv_path, nv_angles, class_type, self.zoperation) #zoom

        """ partial visibility data augmentation  """
        class_type = 'partial_vis'

        pv_angles = [30, 60, 90, 120, 150]
        da_obj.trigger_augmentation(self.pv_aug_data_path, self.pv_path, pv_angles, class_type, self.roperation) #rotation
        pv_aug_data_path_n = os.path.join(self.pv_aug_data_path, self.roperation)
        da_obj.merge_original_rotated_data(pv_aug_data_path_n, self.pv_path)

        pv_angles = [0]
        da_obj.trigger_augmentation(self.pv_aug_data_path, self.pv_path, pv_angles, class_type, self.boperation) #brightness
        da_obj.trigger_augmentation(self.pv_aug_data_path, self.pv_path, pv_angles, class_type, self.zoperation) #zoom

        """ merge aug data with original data """
        self.merge_original_aug_data(da_obj)


    def build_data_pipeline(self, csv_file_path, images_path):

        """ original data reading """
        paths = read_csv_file(csv_file_path, images_path) #extract entries from csv, images path and then separate the data based on labels

        """ read_csv_file creates these three paths and splits the data by reading entries from csv """
        self.fv_path = paths[0]
        self.pv_path = paths[1]
        self.nv_path = paths[2]


        """ apply custom data augmentation to balance the imbalanced data """ #todo can be applied better approaches as mentioned in readme.txt
        da_obj = data_augmentation()
        self.data_augment_merge(da_obj)


        """ Extract sharp images from noisy images """
        """ create paths for single channel images with classes present"""
        fv_single_channel_images_path = '../data/single_channel_images/FULL_VISIBILITY'
        if not os.path.exists(fv_single_channel_images_path):
            os.makedirs(fv_single_channel_images_path)

        pv_single_channel_images_path = '../data/single_channel_images/PARTIAL_VISIBILITY'
        if not os.path.exists(pv_single_channel_images_path):
            os.makedirs(pv_single_channel_images_path)

        nv_single_channel_images_path = '../data/single_channel_images/NO_VISIBILITY'
        if not os.path.exists(nv_single_channel_images_path):
            os.makedirs(nv_single_channel_images_path)

        """Extract single channel images and store them in the created files"""

        """ full_visibility files path """
        extract_single_channel_image(self.fv_path, fv_single_channel_images_path)

        """ partial_visibility files path """
        extract_single_channel_image(self.pv_path, pv_single_channel_images_path)

        """ no_visibility files path """
        extract_single_channel_image(self.nv_path, nv_single_channel_images_path)


        """ Finally split the data as training, validation and testing """
        single_channel_data_path = '../data/single_channel_images'
        sci_tvt_path = '../data/sci_tvt/' # path where training, validation and testing data presents
        ds_obj = data_split(single_channel_data_path)
        ds_obj.split_data(sci_tvt_path)
