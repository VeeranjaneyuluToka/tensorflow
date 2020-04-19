import setGPU

from train import *
from predict import *

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("operation", 'train', "mention whether it is training or prediction")
flags.DEFINE_string('file_path', '', 'give file path if it is prediction')

def main():
    if FLAGS.operation == 'train':
        train()
    elif FLAGS.operation == 'predict':
        predict(FLAGS.file_path)

if __name__=='__main__':
    main()
