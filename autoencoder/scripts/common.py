
import tensorflow as tf

import os
from datetime import date

today = date.today()
d4 = today.strftime("%b-%d-%y")

def model_call_backs(output_dir, filename):
    model_path = os.path.join(output_dir, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    curr_model_path = os.path.join(model_path, d4)
    if not os.path.exists(curr_model_path):
        os.makedirs(curr_model_path)
    modelckpt_path = os.path.join(curr_model_path, filename)

    logs_path = os.path.join(output_dir, 'logs')
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

    return callbacks, curr_model_path

