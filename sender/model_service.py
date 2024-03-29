import math
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pickle
import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, Sequence

# Build and Fit the model

class SimpleGenerator (Sequence) :
    def __init__ (self, data_path, batch_size, model_position, is_lstm) :
        self.data_path = data_path
        self.batch_size = batch_size
        self.model_position = model_position
        self.is_lstm = is_lstm
    
    def __len__(self):
        return math.ceil(75367700/ self.batch_size)

    def __getitem__ (self, idx):
        batch_dataset = pd.read_csv(self.data_path, nrows=self.batch_size, skiprows=idx*self.batch_size, header=None)

        X = batch_dataset.iloc[:, :90].to_numpy().astype(np.float32)
        Y = to_categorical(batch_dataset.iloc[:, 90 - 1 + self.model_position], 43).astype(np.float32)

        if self.is_lstm :
            X = X.reshape(X.shape[0], 1, X.shape[1])

        return X,Y

    
def preprocess_score_to_prob (input_y) :
    # Quality Score Has 0-42 (43 Categorical Class Possible)
    return to_categorical(input_y, 43)

def batch_generator (data_path, batch_size, steps, model_position=None, is_encoded=True) :
    index = 1
    while True :
        data_chunk = pd.read_csv(data_path, skiprows=index*batch_size, nrows=batch_size)

        # Seperate X and Y by Half
        no_of_feature = int(len(data_chunk.columns) / 2)
        x = data_chunk.iloc[:, :no_of_feature]

        # Encode Value
        if is_encoded == False :
            transform_dict = {'A': 1, 'T': 2, 'C' : 3, 'G': 4, 'N': 0}
            x = [transform_dict.get(n,n) for n in x]

        if model_position != None :
            # The Score Position is Selected -> Grab only prefered position and transform
            y = preprocess_score_to_prob(data_chunk.iloc[:, 100 + model_position-1])
        else :
            y - data_chunk.iloc[:, no_of_feature + model_position - 1]

        yield np.array(x, dtype='int8'), np.array(y, dtype='int8')

        if index < steps :
            index += 1
        else :
            index = 1

def single_record_generator (data_path, model_position=1): 
    input_file = open(data_path, 'r')

    while True :
        feature_components = input_file.readline()[:-1].split(',')

        feature_size = len(feature_components)
        x = feature_components[:int(feature_size/2)]
        x = [int (i) for i in x]

        # Encode Value
        # transform_dict = {'A': 1, 'T': 2, 'C' : 3, 'G': 4, 'N': 0}
        # new_x = [int(transform_dict.get(n,n)) for n in x]
        y = to_categorical(feature_components[int(feature_size/2)+ model_position-1], 43)

        yield np.array([x]), np.array([y])

    input_file.close()

def lstm_record_generator (data_path, model_position=1) :
    input_file = open(data_path, 'r')

    while True :
        feature_components = input_file.readline()[:-1].split(',')

        feature_size = len(feature_components)
        x = feature_components[:int(feature_size/2)]
        x = [[int(i) for i in x]]

        new_X = np.array([np.array(x)], dtype=float)

        if model_position != None :
            y = to_categorical(feature_components[int(feature_size/2) + model_position-1], 43)
            yield new_X, np.array([y])
        else :
            y = feature_components[int(feature_size/2):]
            y = [int(i) for i in y]
            yield new_X, np.array([y])
    
    input_file.close()

def lstm_batch_record_generator (data_path, batch_size=200, model_position=1) :
    counter = 1
    input_feature_file = open(data_path, 'r')

    X_container = []
    Y_container = []

    while True:
        if feature_size != 90 :
            print(index_counter, 'Feature Size ', feature_size, line, '\n\n')
            continue
        
        x = feature_components[:feature_size]

        feature_size = math.floor(len(line_component) / 2)

        if len(new_x) > 90 :
            new_x = new_x[:90]
        
        x = np.array(new_x, dtype=np.float32)

        y = to_categorical(feature_components[feature_size + model_position - 1], 43)

        X_container.append(x)
        Y_container.append(y)

        if len(X_container) == batch_size :
            yield np.array([X_container]), np.array(Y_container)
            X_container = []
            Y_container = []

    input_feature_file.close()

def preprocess_data_single_thread (data, model_position) :
    no_of_feature = math.floor(len(data.columns) / 2)

    X = np.array(data.iloc[:, :no_of_feature], dtype=np.float32)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Y = to_categorical(data.iloc[:, no_of_feature + model_position - 1], 43).astype(np.float32)

    return X,Y

def lstm_batch_generator_parallel (data_path, model_position=1, n_cpu_core=8, batch_per_core=200) :
    while True:
        data_chunks = pd.read_csv(data_path, chunksize=n_cpu_core*batch_per_core)

        for chunk in data_chunks :
            no_of_feature = math.floor(len(chunk.columns) / 2)

            X = np.array(chunk.iloc[:, :no_of_feature], dtype=np.float32)
            X = X.reshape(X.shape[0], 1, X.shape[1])
            Y = to_categorical(chunk.iloc[:, no_of_feature + model_position - 1], 43).astype(np.float32)

            yield X,Y

def train_sequencial_model (layers, feature_file, epoch=100, optimiser="adam", loss="categorical_crossentropy", step_per_epoch=20000, model_position=None, is_lstm=False, generator=None, val_generator=None) :
    if generator is not None and val_generator is not None :
        data_batch_generator = generator
        validation_batch_generator = val_generator
    else :
        if is_lstm :
            data_batch_generator = lstm_record_generator(feature_file, model_position=model_position)
            validation_batch_generator = lstm_record_generator(feature_file, model_position=model_position)
        else:
            data_batch_generator = single_record_generator(feature_file, model_position=model_position)
            validation_batch_generator = single_record_generator(feature_file, model_position=model_position)

    model = Sequential(layers)

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])    

    training_hist = model.fit(data_batch_generator, epochs=epoch, steps_per_epoch=step_per_epoch, validation_data=validation_batch_generator, validation_steps=step_per_epoch)

    return model, training_hist