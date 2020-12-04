from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from data_preprocessing import extract_x_y_from_pandas
from utilities import generate_training_statistic_file

import sys
import pickle
import numpy as np
import pandas as pd
import ordinal_categorical_crossentropy as OCC

# Model Memoisation Capacity Experiment
# INPUT: Feature File Path, Destination Hist Path
# OUTPUT: History File

def experiment_builder (data_path, training_hist_folder_path, layers, n_row_per_chunk, n_chunk, epoch, optimiser='sgd', loss='mean_squared_error') :
    dataset = pd.read_csv(data_path, header=None, nrows=n_row_per_chunk * n_chunk)

    for n_data in range(n_row_per_chunk, n_row_per_chunk * n_chunk+1, n_row_per_chunk) :
        X,Y = extract_x_y_from_pandas(dataset.iloc[:n_data, :], model_position=1)
        X = np.array(X, dtype=np.uint8)
        Y = to_categorical(Y, 43).astype(np.uint8)

        model = Sequential(layers)
        model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy', 'mse', 'mae'])
        training_hist = model.fit(x=X, y=Y, epochs=epoch, batch_size=1, validation_data=(X,Y))
        generate_training_statistic_file(training_hist, 'Model_Tester_' + str(n_data), destination_file_path = training_hist_folder_path)

def main (args) :
    layers = Sequential([
        Dense(90, activation='softmax'),
        Dense(90, activation='softmax'),
        Dense(43, activation='softmax'),
    ])
    experiment_builder(args[1], args[2], layers, n_row_per_chunk=10, n_chunk=10, epoch=100, optimiser='adam', loss=OCC.loss)

if __name__ == "__main__":
    main(sys.argv)