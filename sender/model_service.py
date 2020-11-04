import math
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical

# Build and Fit the model

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
    input_file = open(data_path, 'r')

    batch_counter = 0
    X_container = []
    Y_container = []
    index_counter = 0

    while True :
        index_counter += 1
        line = input_file.readline()[:-1]
        feature_components = line.split(',')

        feature_size = math.floor(len(feature_components) / 2)

        if feature_size != 90 :
            print(index_counter, 'Feature Size ', feature_size, line, '\n\n')
        x = feature_components[:feature_size]

        new_x = []
        for x_item in x :
            new_x.append(float(x_item))

        if len(new_x) > 90 :
            new_x = new_x[:90]
        
        x = np.array(new_x, dtype=np.float32)

        y = to_categorical(feature_components[feature_size + model_position - 1], 43)
        # y = [0] * 43
        # y[int(feature_components[feature_size + model_position - 1])-1] = 1
        # y = np.array(y, dtype=np.float32)

        X_container.append(x)
        Y_container.append(y)

        batch_counter += 1

        if len(X_container) == batch_size :
            # print('X!!!!', np.array([X_container]).shape, np.array([X_container]))
            yield np.array([X_container]), np.array(Y_container)
            X_container = []
            Y_container = []
            batch_counter = 0

    input_file.close()

def train_sequencial_model (layers, feature_file, epoch=100, optimiser="adam", loss="categorical_crossentropy", step_per_epoch=20000, model_position=None, is_lstm=False, generator=None) :
    if generator is not None :
        data_batch_generator = generator
        validation_batch_generator = generator
    else :
        if is_lstm :
            data_batch_generator = lstm_record_generator(feature_file, model_position=model_position)
            validation_batch_generator = lstm_record_generator(feature_file, model_position=model_position)
        else:
            data_batch_generator = single_record_generator(feature_file, model_position=model_position)
            validation_batch_generator = single_record_generator(feature_file, model_position=model_position)

    model = Sequential(layers)

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])    

    training_hist = model.fit(data_batch_generator, epochs=epoch, steps_per_epoch=step_per_epoch, validation_data=validation_batch_generator, validation_steps=step_per_epoch, use_multiprocessing=True, workers=8, max_queue_size=100)

    return model, training_hist