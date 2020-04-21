import setGPU

import tensorflow as tf

import numpy as np

from feature_eng import *

model_path = '../artifacts/cnn_class_model.h5'

def predict_from_file(file_path):
    img = get_single_channel_image(file_path)

    scale_img = img * (1./255) # need to scale as we had in training
    ex_dim = np.expand_dims(scale_img, axis=-1);
    ex_img = np.expand_dims(ex_dim, axis=0)

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(ex_img)

    return predictions
    
