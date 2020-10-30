import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical

# Build and Fit the model

def preprocess_score_to_prob (input_y) :
    # Quality Score Has 0-42 (43 Categorical Class Possible)
    return to_categorical(input_y, 43)

def batch_generator (data_path, batch_size, steps, model_position=None) :
    index = 1
    while True :
        data_chunk = pd.read_csv(data_path, skiprows=index*batch_size, nrows=batch_size)

        # Seperate X and Y by Half
        no_of_feature = int(len(data_chunk.columns) / 2)
        x = data_chunk.iloc[:, :no_of_feature]

        # Encode Value
        transform_dict = {'A': 1, 'T': 2, 'C' : 3, 'G': 4, 'N': 0}
        new_x = [transform_dict.get(n,n) for n in x]

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

        y = to_categorical(feature_components[int(feature_size/2) + model_position-1], 43)

        yield new_X, np.array([y])
    
    input_file.close()

def lstm_batch_record_generator (data_path, batch_size=200, model_position=1) :
    input_file = open(data_path, 'r')

    batch_counter = 0
    X_container = []
    Y_container = []

    while True :
        feature_components = input_file.readline()[:-1].split(',')

        feature_size = len(feature_components)
        x = np.array(feature_components(: int(feature_size/2)), dtype=float)

        y = to_categorical(feature_components[int(feature_size/2) + model_position - 1], 43)

        X_container.append(x)
        Y_container.append(y)

        batch_counter += 1

        if batch_counter == batch_size :
            yield np.array(X_container), np.array(Y_container)
            X_container = []
            Y_container = []
            batch_counter = 0

    input_file.close()

def train_sequencial_model (layers, feature_file, epoch=100, optimiser="adam", loss="categorical_crossentropy", step_per_epoch=20000, model_position=None, is_lstm=False, generator=None) :
    # data_batch_generator = batch_generator(feature_file, 256, step_per_epoch, model_position=model_position)

    if generator is not None :
        data_batch_generator = generator
        validation_batch_generator = generator
    else :
        if is_lstm :
            print('Load the data by LSTM')
            data_batch_generator = lstm_record_generator(feature_file,1)
            validation_batch_generator = lstm_record_generator(feature_file,1)
        else:
            print('Load normal set of data')
            data_batch_generator = single_record_generator(feature_file,1)
            validation_batch_generator = single_record_generator(feature_file,1)

    model = Sequential(layers)

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])    

    training_hist = model.fit(data_batch_generator, epochs=epoch, steps_per_epoch=step_per_epoch, validation_data=validation_batch_generator, validation_steps=step_per_epoch)

    return model, training_hist