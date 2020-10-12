import math
import pandas as pd
import numpy as np
import dask.dataframe as dd

from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical

# Build and Fit the model

def preprocess_score_to_prob (input_y) :
    # Quality Score Has 0-42 (43 Categorical Class Possible)
    return to_categorical(input_y, 43)

def batch_generator (data_path, batch_size, steps, model_position=None) :
    index = 1
    dataset = dd.read_csv(data_path, header=None)
    no_of_feature = math.floor(len(dataset.columns) / 2)

    while True :
        # Seperate X and Y
        x = dataset.loc[index].iloc[:,:no_of_feature].compute()

        if model_position is not None :
            y = preprocess_score_to_prob(dataset.loc[index].iloc[:, no_of_feature + model_position - 1].compute())
        else :
            y = dataset.loc[index].iloc[:, no_of_feature:].compute()
        
        yield np.array(x), np.array(y)
        
        if index < steps :
            index += 1
        else :
            index = 1

def train_sequencial_model (layers, feature_file, epoch=100, optimiser="adam", loss="categorical_crossentropy", step_per_epoch=20000, batch_size=10000, model_position=None) :
    data_batch_generator = batch_generator(feature_file, 256, step_per_epoch, model_position=model_position)

    model = Sequential(layers)

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy', 'mse'])    

    training_hist = model.fit(data_batch_generator, epochs=epoch, steps_per_epoch=step_per_epoch, validation_data=data_batch_generator, validation_steps=step_per_epoch)

    return model, training_hist