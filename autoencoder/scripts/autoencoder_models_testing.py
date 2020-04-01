from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator

import os
import scipy.io as sio
import argparse

from cnn_custom_ae_train import custom_generator

class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('ERROR: {}\n\n'.format(message))
        self.print_help()
        sys.exit(2)

"""
load complete auto encoder model
	model_path: path where model file presents
"""
def load_cnn_ae_model(model_path):
    cnn_ae = load_model(model_path)
    return cnn_ae

"""
extract encoder only from auto encoder model
	cnn_ae: complete AE model object
"""
def extract_encoder_from_model(cnn_ae):
    encoder = Model(inputs=cnn_ae.input, outputs=cnn_ae.get_layer('encoder').output)
    return encoder

"""
extract features/bottlenecks from the smaller clip frames using AutoEncoder
	encoder: encoder from AE model
	data_path: data files path
"""
def extract_features(encoder, data_path, fv_folder_name):

    for sub_path in os.listdir(data_path):
        comp_sub_path = data_path + sub_path
        print(comp_sub_path)
        for sub_sub_path in os.listdir(comp_sub_path):
	    #print(sub_sub_path)
            comp_path = comp_sub_path + '/'+ sub_sub_path + '/' + 'frames'
            gen_path = comp_sub_path + '/' + sub_sub_path

            no_of_images = len(os.listdir(comp_path))
            #print(no_of_images)
            nb_batch_size = no_of_images 

            """ use keras generator and call predict_generator """
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(gen_path, target_size =(224,224), color_mode='rgb', batch_size=nb_batch_size)

            file_names = test_generator.filenames
            nb_samples = len(file_names)
            predict_out = encoder.predict_generator(test_generator, steps=int(nb_samples/nb_batch_size))

            """ create folder and save features vectors """
            dst_file_path = gen_path + '/' + fv_folder_name
	    print(dst_file_path)
	    if not os.path.exists(dst_file_path):
	        os.makedirs(dst_file_path)
            fname = sub_sub_path+'.mat'
            dst_fin_file_path = dst_file_path + '/' + fname
	    #print(dst_fin_file_path)
	    sio.savemat(dst_fin_file_path, {'logits':predict_out})

"""
extract features using autoencoder by giving all the data once
	encder: encoder from auto encoder
"""
def extract_features_frames(encoder):
    data_path = '/mnt/disks/slow1/video_processing/exp/AE_arch/data/movie_data/train_bbs/'
    nb_batch_size = 256
    nb_samples = len(os.listdir(data_path+'frames'))
	
    #test_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = custom_generator(data_path, nb_samples, nb_batch_size, 128)
    test_generator = test_datagen.flow_from_directory(data_path, target_size=(128,128), color_mode='rgb', batch_size=nb_batch_size)
    predict_out = encoder.predict_generator(test_generator, steps=int(nb_samples/nb_batch_size))
    dst_file_path = data_path + 'feature_vectors_ae'
    if not os.path.exists(dst_file_path):
	os.makedirs(dst_file_path)
    fname = 'bbs_feature_vectors_ae.mat'
    fin_dst_file_path = dst_file_path + '/' + fname
    sio.savemat(fin_dst_file_path, {'logits':predict_out})


def main():
    parser = ArgParser(description="Extract feature vector from the AutoEncoder model")
    required_args = parser.add_argument_group("Mandatory arguments")
    required_args.add_argument("--model_path", dest="model_path", type=str, required=True, help="path of the autoencoder model")
    required_args.add_argument("--data_path", dest = "data_path", type=str, required=True, help="path where data presents, path should be three level up from frames")
    required_args.add_argument("--fv_folder_name", dest="fv_folder_name", type=str, required=True, help="feature vector(FV) folder name in which file will be stored")
    args = parser.parse_args()

    """ load model and extract encoder from complete autoencder model """
    cnn_ae = load_cnn_ae_model(args.model_path)
    encoder = extract_encoder_from_model(cnn_ae)

    """ extract features using auto encoder """
    print(args.fv_folder_name)
    extract_features(encoder, args.data_path, args.fv_folder_name)
	
    """ extract features using auto encoder by directly feeding all the images once """
    #extract_features(encoder)

if __name__ == "__main__":
    main()
