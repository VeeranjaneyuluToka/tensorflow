import setGPU

from tensorflow.keras.datasets import cifar10

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from kerastuner.tuners import (BayesianOptimization, Hyperband, RandomSearch)
from kerastuner import HyperModel

import tensorflow as tf

from pathlib import Path
from loguru import logger
import time

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
SEED = 1
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
HYPERBAND_MAX_EPOCHS = 40
BAYESIAN_NUM_INITIAL_POINTS = 1
N_EPOCH_SEARCH = 40

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")/255.0
    x_test = x_test.astype("float32")/255.0

    return x_test, x_train, y_test, y_train

def set_gpu_config():
    # Set up GPU config
    logger.info("Setting up GPU if found")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build(self, hp):
        model = keras.Sequential()

        model.add(Conv2D(filters = 16, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(filters=16, activation='relu', kernel_size=3))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05,)))
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        model.add(Conv2D(filters=hp.Choice("num_filters", values=[32, 64], default=64,), activation='relu', kernel_size=3))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(rate=hp.Float("dropout_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05,)))
        model.add(Flatten())
        model.add(Dense(units=hp.Int("units", min_value=32, max_value=512, step=32, default=128), activation=
                        hp.Choice("dense_activation", values=['relu', 'tanh', "sigmoid"], default='relu'),))
        model.add(Dropout(rate=hp.Float("dropout_3", min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(optimizer=keras.optimizers.Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)),
                     loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        return model


def define_tuners(hypermodel, directory, project_name):
    random_tuner = RandomSearch(hypermodel, objective='val_acc', seed=SEED, max_trials=MAX_TRIALS, executions_per_trial=EXECUTION_PER_TRIAL,
                                directory=f"{directory}_random_search", project_name=project_name)
    hyperband_tuner = Hyperband(hypermodel, max_epochs=HYPERBAND_MAX_EPOCHS, objective="val_acc", seed=SEED, executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_hyperband", project_name=project_name,)

    bayesian_tuner = BayesianOptimization(hypermodel, objective='val_acc',seed=SEED, num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,
        max_trials=MAX_TRIALS, directory=f"{directory}_bayesian", project_name=project_name)
    return [random_tuner, hyperband_tuner, bayesian_tuner]

def tuner_evaluation(tuner, x_test, x_train, y_test, y_train):
    set_gpu_config()

    # Overview of the task
    tuner.search_space_summary()

    # Performs the hyperparameter tuning
    logger.info("Start hyperparameter tuning")
    search_start = time.time()
    tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)
    search_end = time.time()
    elapsed_time = search_end - search_start

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(x_test, y_test)

    return elapsed_time, loss, accuracy


def run_hyper_parameter_training():
    x_test, x_train, y_test, y_train = load_data()

    #print(x_test.shape, x_train.shape, y_test.shape, y_train.shape)
    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes = NUM_CLASSES)
    output_dir = Path("../../outputs/cifar10/")

    tuners = define_tuners(hypermodel, directory=output_dir, project_name="simple_cnn_tuning")

    results = []
    for tuner in tuners:
        elapsed_time, loss, accuracy = tuner_evaluation(tuner, x_test, x_train, y_test, y_train)
        logger.info(f"Elapsed time = {elapsed_time:10.4f} s, accuracy = {accuracy}, loss = {loss}")
        results.append([elapsed_time, loss, accuracy])
    logger.info(results)


def main():
    run_hyper_parameter_training()

if __name__ == "__main__":
    main()
