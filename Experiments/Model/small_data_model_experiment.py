from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from data_preprocessing import extract_x_y_from_pandas
from utilities import generate_training_statistic_file

import sys
import pickle
import numpy as np
import pandas as pd
import ordinal_categorical_crossentropy as OCC

# Model Memoisation Capacity Experiment
# INPUT: Feature File Path, Destination Hist Path, Model Path, Experiment Name
# OUTPUT: History File

def learning_rate_scheduler (epoch, lr) :
    if epoch == 100 :
        lr = lr/2
    return lr

def experiment_builder (data_path, training_hist_folder_path, model_path, layers, n_row_per_chunk, n_chunk, epoch, optimiser='sgd', loss='mean_squared_error', experiment_name='Base_Model', callbacks=[]) :
    dataset = pd.read_csv(data_path, header=None, nrows=n_row_per_chunk * n_chunk)

    # for n_data in range(n_row_per_chunk, n_row_per_chunk * n_chunk, n_row_per_chunk) :
    for n_data in range(1000,1001,1000):
        X,Y = extract_x_y_from_pandas(dataset.iloc[:n_data, :], model_position=1)
        X = np.array(X, dtype=np.uint8)
        Y = to_categorical(Y, 43).astype(np.uint8)

        model = Sequential()

        for layer in layers :
            model.add(layer)

        model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy', 'mse', 'mae'])
        training_hist = model.fit(x=X, y=Y, epochs=epoch, batch_size=1, validation_data=(X,Y), callbacks=callbacks)
        generate_training_statistic_file(training_hist, experiment_name + '_' + str(n_data), destination_file_path = training_hist_folder_path)
        model.save(model_path + '/' + experiment_name + '_' + str(n_data) + '.h5')

    # for n_data in range(10000,100000,10000):
    #     X,Y = extract_x_y_from_pandas(dataset.iloc[:n_data, :], model_position=1)
    #     X = np.array(X, dtype=np.uint8)
    #     Y = to_categorical(Y, 43).astype(np.uint8)

    #     model = Sequential()

    #     for layer in layers :
    #         model.add(layer)

    #     model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy', 'mse', 'mae'])
    #     training_hist = model.fit(x=X, y=Y, epochs=epoch, batch_size=1, validation_data=(X,Y))
    #     generate_training_statistic_file(training_hist, 'Model_Tester_' + str(n_data), destination_file_path = training_hist_folder_path)

    #     model.save('Results/model_experiment/model/Base_Model/Model_Tester_' + str(n_data) + '.h5')

def main (args) :
    layers = [
        Dense(90, activation='softmax'),
        Dense(90, activation='softmax'),
        Dense(43, activation='softmax'),
    ]
    optimiser = Adam()
    loss = OCC.loss
    lrs_callback = LearningRateScheduler(learning_rate_scheduler)

    experiment_builder(args[1], args[2], args[3], layers, n_row_per_chunk=1000, n_chunk=100, epoch=200, optimiser=optimiser, loss=loss, experiment_name=args[4], callbacks=[lrs_callback])

if __name__ == "__main__":
    main(sys.argv)