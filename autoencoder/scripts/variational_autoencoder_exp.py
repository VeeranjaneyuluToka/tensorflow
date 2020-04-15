import setGPU

import tensorflow as tf

import os
import cv2 as cv
import numpy as np

from common import *
from data_reading import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("training_data_path", "/mnt/data/Veeru_backup/cv_exp/data/movie_titles/mini_data/train/", "pass the training data path")
flags.DEFINE_string("testing_data_path", "/mnt/data/Veeru_backup/cv_exp/data/movie_titles/mini_data/validation/", "pass the testing data path")
flags.DEFINE_string("output_path", "/mnt/data/Veeru_backup/cv_exp/src/AE/outputs/", "outputs path")

class inference_deep_vae(object):
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.img_chns = 3

        self.filters = 64
        self.num_conv = 3
        self.batch_size = 64

        if tf.keras.backend.image_data_format() == 'channels_first':
            original_img_size = (img_chns, img_rows, img_cols)
        else:
            original_img_size = (img_rows, img_cols, img_chns)

        self.latent_dim = 2
        self.intermediate_dim = 128
        self.epsilon_std = 1.0
        self.epochs = 50

    def inference(self):

        x = tf.keras.layers.Input(shape=self.original_img_size)

        conv_1 = tf.keras.layers.Conv2D(self.img_chns, kernel_size=(2, 2), padding='same', activation='relu')(x)
        conv_2 = tf.keras.layers.Conv2D(self.filters, kernel_size=(2, 2), padding='same', activation='relu', strides=(2, 2))(conv_1)
        conv_3 = tf.keras.layers.Conv2D(self.filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_2)
        conv_4 = tf.keras.layers.Conv2D(self.filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_3)
        flat = tf.keras.layers.Flatten()(conv_4)
        hidden = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')(flat)

        z_mean = tf.keras.layers.Dense(self.latent_dim)(hidden)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(hidden)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
            return z_mean + tf.keras.backend.exp(z_log_var) * epsilon

        z = tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        decoder_hid = tf.keras.layers.Dense(self.intermediate_dim, activation='relu')
        decoder_upsample = tf.keras.layers.Dense(self.filters * 128 * 128, activation='relu')

        if tf.keras.backend.image_data_format() == 'channels_first':
            output_shape = (self.batch_size, self.filters, 128, 128)
        else:
            output_shape = (self.batch_size, 128, 128, self.filters)

        decoder_reshape = tf.keras.layers.Reshape(output_shape[1:])
        decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(self.filters, kernel_size=self.num_conv, padding='same', strides=1, activation='relu')
        decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(self.filters, kernel_size=self.num_conv, padding='same', strides=1, activation='relu')

        if tf.keras.backend.image_data_format() == 'channels_first':
            output_shape = (self.batch_size, self.filters, 256, 256)
        else:
            output_shape = (self.batch_size, 256, 256, self.filters)

        decoder_deconv_3_upsamp = tf.keras.layers.Conv2DTranspose(self.filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
        decoder_mean_squash = tf.keras.layers.Conv2D(self.img_chns, kernel_size=2, padding='valid', activation='sigmoid')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        #instantiate VAE model
        vae = tf.keras.models.Model(x, x_decoded_mean_squash)
        #tf.keras.utils.plot_model(vae, to_file='../model/vae_md/vae_model_md_040219.png', show_shapes=True)

        return vae, z_log_var, z_mean

#vae, z_log_var, z_mean = inference()

class inference_vae(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.input_shape = (img_size, img_size, 3)
        self.batch_size = 64
        self.kernel_size = 3
        self.filters = 16
        self.latent_dim = 2
        self.epochs = 30
        self.z_mean = 0
        self.z_log_var = 0

    """
    reparameterization trick
    instead of sampling from Q(z/x), sample eps = N(0, I)
    then z = z_mean+sqrt(var)*eps
    """
    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        batch = tf.keras.backend.shape(self.z_mean)[0]
        dim = tf.keras.backend.int_shape(self.z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch,dim))

        return self.z_mean + tf.keras.backend.exp(0.5*self.z_log_var) * epsilon

    """ build encoder architecture """
    def encoder_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            self.filters *=2
            x = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(x)

        shape = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        self.z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        self.z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = tf.keras.layers.Lambda(self.sampling, output_shape=(self.latent_dim, ), name='z')([self.z_mean, self.z_log_var])

        encoder = tf.keras.models.Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')

        #encoder.summary()
        #tf.keras.utils.plot_model(encoder, to_file='../model/vae_mnist/vae_cnn_encoder_md.png', show_shapes=True)

        #return encoder, shape, inputs, z_mean, z_log_var
        return encoder, shape, inputs

    """ build decoder architecture """
    def decoder_model(self, shape, encoder, inputs):
        latent_inputs = tf.keras.layers.Input(shape=(self.latent_dim, ), name='z_sampling')
        x = tf.keras.layers.Dense(shape[1]*shape[2]*shape[3], activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(2):
            x = tf.keras.layers.Conv2DTranspose(filters=self.filters, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(x)
            self.filters //= 2

        outputs = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=self.kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)

        decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

        #decoder.summary()
        #tf.keras.utils.plot_model(decoder, to_file='../model/vae_mnist/vae_cnn_decoder_md.png', show_shapes=True)

        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.models.Model(inputs, outputs, name='vae')

        return decoder, vae, outputs

    def loss_fun(self, inputs, outputs, loss_type='mse'):
        if loss_type == 'mse':
            recont_loss = tf.keras.losses.mse(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(outputs))
        else:
            recont_loss = tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten())
        recont_loss *= self.img_size*self.img_size
        kl_loss = 1 + self.z_log_var - tf.keras.backend.square(self.z_mean) - tf.keras.backend.exp(self.z_log_var)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(recont_loss+kl_loss)

        return vae_loss


    def train_min_samples(self, encoder, decoder, vae, inputs, outputs, x_train, x_test):
        models = (encoder, decoder)

        vae.compile(optimizer='rmsprop', loss = self.loss_fun, metrics=['acc'])

        #tf.keras.utils.plot_model(vae, to_file='../model/vae_mnist/vae_cnn_md.png', show_shapes=True)

        callbacks, curr_model_path = model_call_backs(FLAGS.output_path, 'vae_model.h5')
        vae.fit(x_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, None), callbacks=callbacks)

        model_path = os.path.join(curr_model_path, 'vae_file_model.h5')
        vae.save(model_path)

    def train_generator(self, encoder, decoder, inputs, outputs, vae):
        models = (encoder, decoder)
        vae.compile(optimizer='rmsprop', loss=self.loss_fun, metrics=['acc'])

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_format='channels_last', rescale=1./255)
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_format='channels_last', rescale=1./255)

        train_generator = train_datagen.flow_from_directory(FLAGS.training_data_path, color_mode='rgb', class_mode='input', batch_size=self.batch_size)
        valid_generator = test_datagen.flow_from_directory(FLAGS.testing_data_path, color_mode='rgb', class_mode='input', batch_size=self.batch_size)

        #tf.keras.utils.plot_model(vae, to_file='../model/vae_mnist/vae_cnn_mdn.png', show_shapes=True)

        train_samples = train_generator.n//self.batch_size
        valid_samples = valid_generator.n//self.batch_size

        callbacks, curr_model_path = model_call_backs(FLAGS.output_path, 'vae_model.h5')
        vae.fit_generator(train_generator, steps_per_epoch= train_samples, epochs=self.epochs, validation_data=valid_generator, validation_steps = valid_samples, 
                callbacks = callbacks, workers=16, use_multiprocessing=True)

        model_path = os.path.join(curr_model_path, 'vae_final_model.h5')
        vae.save(model_path)

def main():
    is_fit = False

    img_size = 256
    if is_fit: #experiment with small dataset which can fit in available memory
        data = data_reading();
        img_size, x_train, x_test = data.create_data_numpy_arrays(FLAGS.training_data_path, FLAGS.testing_data_path)

    arch = inference_vae(img_size)
    encoder, shape, inputs = arch.encoder_model()
    decoder, vae, outputs = arch.decoder_model(shape, encoder, inputs)

    if is_fit:
        arch.train_min_samples(encoder, decoder, vae, inputs, outputs, x_train, x_test)

    arch.train_generator(encoder, decoder, inputs, outputs, vae)

if __name__ == "__main__":
    main()

