import setGPU

import tensorflow as tf

from autoencoder_architectures import *

import os
import cv2 as cv
import math
import numpy as np
from datetime import date

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data_dir', '/mnt/data/Veeru_backup/cv_exp/data/movie_titles/mini_data/train/', 'training data directory path')
flags.DEFINE_string('valid_data_dir', '/mnt/data/Veeru_backup/cv_exp/data/movie_titles/mini_data/validation/', 'testing data directory path')
flags.DEFINE_string('output_dir', '../../outputs/', 'all outputs model, logs etc.. path')

today = date.today()
d4 = today.strftime("%b-%d-%y")

"""
class derived from keras.utils.Sequence class, so that the data will be processed sequentially
"""
class custom_generator(tf.keras.utils.Sequence):
    def __init__(self, folder, image_filenames, batch_size, image_size):
        self.folder = folder
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.image_size = image_size

        """ Initialize the Keras ImageDataGenerator to augment the data """
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True, zca_whitening=False, 
                rotation_range=90, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)

    def __load__(self, filename):
        img_path = os.path.join(self.folder, filename)
        image = cv.imread(img_path)
        res_image = cv.resize(image, (self.image_size, self.image_size))
        res_image = res_image / 255.
        return res_image

    def get_file_names(self):
        return self.image_files
        
    def __len__(self):
        return int(np.ceil(len(self.image_filenames)/float(self.batch_size)))

    def __normalize__(self, image):
        """
        call ImageDataGenerator fit() method on each batch image to get statistics of batch
        """
        self.datagen.fit(image)
        #self.datagen.standardize(image)
    
    def __getitem__(self, idx):
        if (idx+1)*self.batch_size > len(self.image_filenames):
            self.batch_size = len(self.image_filenames) - (idx*self.batch_size)

        batch_x = self.image_filenames[idx*self.batch_size:(idx+1)*self.batch_size]

        img_lst = []
        for filename in batch_x:
            tmp_img = self.__load__(filename)
            img_lst.append(tmp_img)
            
        train_image = np.array(img_lst)
        #self.__normalize__(image)

        noise_factor = 0.3
        train_image_noisy = train_image + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=train_image.shape)
        self.__normalize__(train_image_noisy)

        return train_image, train_image

    def on_epoch_end(self):
        pass

class fixedImageDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def standardize(self, x):
        width, height, channel_axis = x.shape
        print(x.shape[channel_axis-1])
        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[channel_axis - 2] = x.shape[channel_axis-1]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

class dataGenerator:
    def __init__(self, train_data_dir, valid_data_dir):
        self.img_width = 224
        self.img_height = 224
        self.img_channels = 3
        self.nb_epochs = 1000
        self.nb_batch_size = 32
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
	
    def fixed_generator(self, generator):
        for batch in generator:
            yield(batch, batch)


    def feed_data_train_idg(self, ae):
        #add the data augmentation
        shift = 0.2

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=shift, zoom_range=shift, rotation_range=90, width_shift_range=shift, height_shift_range=shift, 
                horizontal_flip=True, vertical_flip=True)
        valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=shift, zoom_range=shift, rotation_range=90, width_shift_range=shift, height_shift_range=shift, 
                horizontal_flip=True, vertical_flip=True)
        train_generator = train_datagen.flow_from_directory(self.train_data_dir, target_size=(self.img_width, self.img_height), batch_size=self.nb_batch_size, class_mode="input")
        valid_generator = valid_datagen.flow_from_directory(self.valid_data_dir, target_size=(self.img_width, self.img_height), batch_size=self.nb_batch_size, class_mode="input")

        ae.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss = tf.keras.losses.mean_squared_error, metrics=['acc'])

        train_samples = int(train_generator.n//self.nb_batch_size)
        valid_samples = int(valid_generator.n//self.nb_batch_size)

        model_path = os.path.join(FLAGS.output_dir, 'model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        curr_model_path = os.path.join(model_path, d4)
        if not os.path.exists(curr_model_path):
            os.makedirs(curr_model_path)
        modelckpt_path = os.path.join(curr_model_path, 'aecnn.h5')

        logs_path = os.path.join(FLAGS.output_dir, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        curr_logs_path = os.path.join(logs_path, d4)
        if not os.path.exists(curr_logs_path):
            os.makedirs(curr_logs_path)
        comp_logs_path = os.path.join(curr_logs_path, 'tb')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=modelckpt_path, save_best_only=True, monitor='val_loss', mode='min')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=comp_logs_path, histogram_freq=0, write_graph=False)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', verbose=1)
        callbacks = [checkpoint, tensorboard, es]

        data = train_generator.next()
        #kmri.visualize_model(ae, data[0])

        ae.fit_generator(train_generator,steps_per_epoch=train_samples,epochs=self.nb_epochs, validation_data=valid_generator, validation_steps=valid_samples, 
                callbacks=callbacks, workers=32, use_multiprocessing=True)

        final_model = os.path.join(curr_model_path, 'aecnn_MD_final.h5')
        ae.save(final_model)

    def feed_train_cdg(self, ae, train_filenames, valid_filenames):
        train_datagen = custom_generator(FLAGS.train_data_dir, train_filenames, self.nb_batch_size, self.img_width)
        valid_datagen = custom_generator(FLAGS.valid_data_dir, valid_filenames, self.nb_batch_size, self.img_width)
		
        ae.compile(optimizer=optimizers.Adam(0.01), loss = losses.mean_squared_error, metrics=['accuracy'])

        nb_train_steps = int(train_datagen.n // self.nb_batch_size)
        nb_valid_steps = int(valid_datagen.n // self.nb_batch_size)

        model_path = os.path.join(FLAGS.output_dir, 'model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        curr_model_path = os.path.join(model_path, d4)
        if not os.path.exists(curr_model_path):
            os.makedirs(curr_model_path)
        modelckpt_path = os.path.join(curr_model_path, 'aecnn.h5')

        logs_path = os.path.join(FLAGS.output_dir, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        curr_logs_path = os.path.join(logs_path, d4)
        if not os.path.exists(curr_logs_path):
            os.makedirs(curr_logs_path)
        comp_logs_path = os.path.join(curr_logs_path, 'tb')

        checkpoint = ModelCheckpoint(filepath=modelckpt_path, save_best_only=True, monitor='val_loss', mode='min')
        tensorboard = TensorBoard(log_dir=comp_logs_path, histogram_freq=0, write_graph=False)
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', verbose=1)

        ae.fit_generator(generator=train_datagen, steps_per_epoch=nb_train_steps, epochs=self.nb_epochs, validation_data=valid_datagen, validation_steps = nb_valid_steps,
		callbacks=[checkpoint, tensorboard, es], workers=32, use_multiprocessing=True)

        final_model = os.path.join(curr_model_path, 'aeCNN_final.h5')
        ae.save(final_model)

def train(train_data_dir, valid_data_dir, train_cdg):

    dg = dataGenerator(train_data_dir, valid_data_dir)

    input_shape = (224, 224, 3)
	
    """ autoencoder with inception like architecture """
    #ae_model = inceptionAutoencoder(input_shape).autoencoder
    #print(ae_model.summary())

    """ autoencoder with vgg16 like architecture by providing pre-trained weights """
    #model_path = '/mnt/disks/slow1/video_processing/exp/AE_arch/model/vgg16_model/vgg16_weights_notop.h5'
    #ae_model = VGG16AutoEncoder(input_shape, model_path, 5).auto_encoder
    #print(ae_model.summary())

    """ Convolutional autoencoder architecture """
    ae_model = ConvAutoEncoder(input_shape).autoencoder
    print(ae_model.summary())

    if train_cdg == True: # use custom generator
        train_filenames = os.listdir(train_data_dir+'frames/')
        valid_filenames = os.listdir(valid_data_dir+'frames/')
        dg.feed_train_cdg(ae_model, train_filenames, valid_filenames)

    else: # use keras ImageDataGenerator
        dg.feed_data_train_idg(ae_model)

def main():
    train_cdg = False #Enable custom generator
    train(FLAGS.train_data_dir, FLAGS.valid_data_dir, train_cdg)	

if __name__ == "__main__":
    main()
