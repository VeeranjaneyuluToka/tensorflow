import setGPU

from train import *
from predict import *

import tensorflow as tf

import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("operation", 'train', "mention whether it is training or prediction")
flags.DEFINE_string("csv_file_path", '../data/gicsd_labels.csv', "csv file path")
flags.DEFINE_string("images_path", '../data/images', "images path")

flags.DEFINE_string('file_path', '', 'give file path if it is prediction')

np.set_printoptions(formatter={'float_kind':'{:f}'.format}) # to display the prediction in non-scientific values

def train():
    dp_obj = data_pipeline()
    dp_obj.build_data_pipeline(FLAGS.csv_file_path, FLAGS.images_path)

    tn_obj = train_network()
    tn_obj.train()

def main():
    if FLAGS.operation == 'train':
        train()
    elif FLAGS.operation == 'predict':
        print(predict_from_file(FLAGS.file_path))

if __name__=='__main__':
    main()
