import setGPU

import tensorflow as tf

from autoencoder_architectures import *

import os
import cv2 as cv
import math
import numpy as np

from common import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data_dir', '/mnt/data/Veeru_backup/cv_exp/data/movie_titles/mini_data/train/', 'training data directory path')
flags.DEFINE_string('valid_data_dir', '/mnt/data/Veeru_backup/cv_exp/data/movie_titles/mini_data/validation/', 'testing data directory path')
flags.DEFINE_string('output_dir', '../../outputs/', 'path where all outputs i.e.. model, logs etc.. stored')

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
        self.nb_epochs = 5
        self.nb_batch_size = 32
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
	
    def fixed_generator(self, generator):
        for batch in generator:
            yield(batch, batch)

    """
    Demo code of using ImageDataGenerator().fit method
    limitations: need to store all the data in ram which is not feasible in larger data cases
    """
    def feed_data_train_fit_transform(ae):

        train_datagen_samples = tf.keras.preprocessing.image.ImageDataGenerator()
        valid_datagen_samples = tf.keras.preprocessing.image.ImageDataGenerator()

        train_generator_samples = train_datagen_samples.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                                        batch_size=nb_batch_size, class_mode="input", shuffle=True)
        valid_generator_samples = valid_datagen_samples.flow_from_directory(valid_data_dir, target_size=(img_width, img_height),
                                                                        batch_size=nb_batch_size, class_mode="input", shuffle=True)

        #train samples
        fit_train_samples = np.array([])
        fit_train_samples.resize((0, img_width, img_height, 3))
        for i in range(int(train_generator_samples.samples/nb_batch_size)):
            imgs, labels = next(train_generator_samples)
            idx = np.random.choice(imgs.shape[0], nb_batch_size, replace=False)
            np.vstack((fit_train_samples, imgs))

        if train_generator_samples.samples % nb_batch_size != 0:
            imgs, labels = next(train_generator_samples)
            idx = np.random.choice(imgs.shape[0], nb_batch_size, replace=False)
            np.vstack((fit_train_samples, imgs))

        #validation samples
        fit_valid_samples = np.array([])
        fit_valid_samples.resize((0, img_width, img_height, 3))
        for i in range(int(valid_generator_samples.samples/nb_batch_size)):
            imgs, labels = next(valid_generator_samples)
            idx = np.random.choice(imgs.shape[0], nb_batch_size, replace=False)
            np.vstack((fit_valid_samples, imgs[idx]))

        if valid_generator_samples.samples % nb_batch_size != 0:
            imgs, labels = next(valid_generator_samples)
            idx = np.random.choice(imgs.shape[0], imgs.shape[0], replace=False)
            np.vstack((fit_valid_samples, imgs[idx]))

        shift = 0.2
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True, zca_whitening=False,
                                       rotation_range=90, width_shift_range=shift, height_shift_range=shift, horizontal_flip=True, vertical_flip=True)
        valid_datagen = tf.keras.preprocessing.ImageDataGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=False, zca_whitening=False,
                                       rotation_range=90,width_shift_range=shift, height_shift_range=shift, horizontal_flip=True, vertical_flip=True)

        train_datagen.fit(fit_train_samples)
        valid_datagen.fit(fit_valid_samples)

        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=nb_batch_size, class_mode="input",
                                                        shuffle=True)
        valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size=(img_width, img_height), batch_size=nb_batch_size, class_mode="input",
                                                        shuffle=True)

        train_samples = int(train_generator.n//self.batch_size)
        valid_samples = int(valid_generator.n//self.batch_size)
        ae.fit_generator(train_generator, steps_per_epoch = train_samples, validation_data=valid_generator, validation_steps = valid_samples, epochs=nb_epoch)

        ae.save("../model/ae_movie_data.h5")
        del ae

    def feed_data_train_idg(self, ae, arch_type):
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

        callbacks, curr_model_path = model_call_backs(FLAGS.output_dir, arch_type+'_MD.h5')

        data = train_generator.next()
        #kmri.visualize_model(ae, data[0])

        ae.fit_generator(train_generator,steps_per_epoch=train_samples,epochs=self.nb_epochs, validation_data=valid_generator, validation_steps=valid_samples, 
                callbacks=callbacks, workers=32, use_multiprocessing=True)

        final_model = os.path.join(curr_model_path, arch_type+'_MD_final.h5')
        ae.save(final_model)

    def feed_train_cdg(self, ae, train_filenames, valid_filenames, arch_type):
        train_datagen = custom_generator(FLAGS.train_data_dir, train_filenames, self.nb_batch_size, self.img_width)
        valid_datagen = custom_generator(FLAGS.valid_data_dir, valid_filenames, self.nb_batch_size, self.img_width)
		
        ae.compile(optimizer=optimizers.Adam(0.01), loss = losses.mean_squared_error, metrics=['accuracy'])

        nb_train_steps = int(train_datagen.n // self.nb_batch_size)
        nb_valid_steps = int(valid_datagen.n // self.nb_batch_size)

        callbacks, curr_model_path = model_call_backs(FLAGS.output_dir, arch_type+'_MD.h5')

        ae.fit_generator(generator=train_datagen, steps_per_epoch=nb_train_steps, epochs=self.nb_epochs, validation_data=valid_datagen, validation_steps = nb_valid_steps,
		callbacks=callbacks, workers=32, use_multiprocessing=True)

        final_model = os.path.join(curr_model_path, arch_type+'_MD_final.h5')
        ae.save(final_model)

def train(train_data_dir, valid_data_dir, train_cdg, arch_type):

    dg = dataGenerator(train_data_dir, valid_data_dir)

    input_shape = (224, 224, 3)
	
    """ autoencoder with inception like architecture """
    if arch_type == 'CNN_INC_AE':
        ae_model = inceptionAutoencoder(input_shape).autoencoder

    """ autoencoder with vgg16 like architecture by providing pre-trained weights """
    if arch_type == 'CNN_VGG_AE':
        model_path = '/mnt/data/Veeru_backup/gcp_backup/pre_trained_models/vgg16_weights_notop.h5'
        ae_model = VGG16AutoEncoder(input_shape, model_path, 5).auto_encoder

    """ Convolutional autoencoder architecture """
    if arch_type == 'CNN_AE':
        is_deeper = True
        if is_deeper: 
            ae_model = ConvAutoEncoder(input_shape, is_deeper).autoencoder
        else:
            ae_model = ConvAutoEncoder(input_shape).ae_4layers()

    if arch_type == 'CNN_UNET_AE':
        ae_model = UnetAutoEncoder(input_shape).inference_UNET_1()

    print(ae_model.summary())

    if train_cdg == True: # use custom generator
        train_filenames = os.listdir(train_data_dir+'frames/')
        valid_filenames = os.listdir(valid_data_dir+'frames/')
        dg.feed_train_cdg(ae_model, train_filenames, valid_filenames, arch_type)

    else: # use keras ImageDataGenerator
        dg.feed_data_train_idg(ae_model, arch_type)

def main():
    train_cdg = False #Enable custom generator
    arch_type = 'CNN_UNET_AE'
    train(FLAGS.train_data_dir, FLAGS.valid_data_dir, train_cdg, arch_type)	

if __name__ == "__main__":
    main()
