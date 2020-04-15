import setGPU

import tensorflow as tf

import os
import scipy.io as sio

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", "/mnt/data/Veeru_backup/cv_exp/data/movie_titles/ae_models_test_data/",  'path where data presents')
flags.DEFINE_string("model_path", "/mnt/data/Veeru_backup/cv_exp/src/AE/outputs/model/Apr-03-20/aecnn_MD_final.h5", 'path where model presents')

class feature_extraction(object):
    def __init__(self):
        self.batch_size = 32
        self.target_size = 224

    """
    load complete auto encoder model
	model_path: path where model file presents
    """
    def load_ae_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    """
    extract encoder only from auto encoder model
	cnn_ae: complete AE model object
    """
    def extract_encoder_from_model(self, model):
        encoder = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
        return encoder

    """
    extract features/bottlenecks from the smaller clip frames using AutoEncoder
	encoder: encoder from AE model
	data_path: data files path where one level up from images
    """
    def extract_features_idg(self, encoder, data_path):

        """ use keras generator and call predict_generator """
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(data_path, target_size =(self.target_size, self.target_size), color_mode='rgb', batch_size=self.batch_size)

        file_names = test_generator.filenames

        remainder = test_generator.n%self.batch_size
        if remainder != 0:
            samples_per_batch = (test_generator.n//self.batch_size)+1
        else:
            samples_per_batch = (test_generator.n//self.batch_size)
        predict_out = encoder.predict_generator(test_generator, steps=int(samples_per_batch))

        """ create folder and save features vectors
        dst_file_path = os.path.join(data_path, fv_folder_name)
        if not os.path.exists(dst_file_path):
            os.makedirs(dst_file_path)
        fname = sub_sub_path+'.mat'
        dst_fin_file_path = os.path.join(dst_file_path, fname)
        sio.savemat(dst_fin_file_path, {'logits':predict_out})"""

        return predict_out, file_names

    """
    extract features using autoencoder by giving all the data once
	encder: encoder from auto encoder
    """
    def extract_features_cg(self, encoder, data_path):
        nb_samples = len(os.listdir(data_path))
	
        test_datagen = custom_generator(data_path, nb_samples, self.batch_size, self.target_size)
        test_generator = test_datagen.flow_from_directory(data_path, target_size=(self.target_size, self.target_size), color_mode='rgb', batch_size=self.batch_size)

        predict_out = encoder.predict_generator(test_generator, steps=int(nb_samples/self.batch_size))

        dst_file_path = os.path.join(data_path, 'feature_vectors_ae')
        if not os.path.exists(dst_file_path):
            os.makedirs(dst_file_path)
        fin_dst_file_path = os.path.join(dst_file_path, 'feature_vectors_ae.mat')
        sio.savemat(fin_dst_file_path, {'logits':predict_out})

def main():

    is_cg = False #select the generator between tf.keras.preprocessing.image.ImageDataGenerator or tf.keras.utils.Sequence

    feat_ext = feature_extraction()

    """ load model and extract encoder from complete autoencder model """
    model = feat_ext.load_cnn_ae_model(FLAGS.model_path)
    encoder_model = feat_ext.extract_encoder_from_model(model)

    if is_cg == False:
        """ extract features using auto encoder model """
        feat_ext.extract_features_idg(encoder_model, FLAGS.data_path)
	
    else:
        """ extract features using auto encoder by directly feeding all the images once """
        extract_features_cg(encoder_model, FLAGS.data_path)

if __name__ == "__main__":
    main()
