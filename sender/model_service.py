import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential 
from tensorflow.keras.utils import to_categorical

# Build and Fit the model

def preprocess_score_to_prob (input_y) :
    # Quality Score Has 0-42 (43 Categorical Class Possible)
    return to_categorical(input_y, 43)

def load_data_as_batch (data_path, index, batch_size, model_position=None) :
    dataset = pd.read_csv(data_path, skiprows=index*batch_size, nrows=batch_size)

    no_of_feature = int(len(dataset.columns) / 2)

    # Seperate X and Y by Half
    x = dataset.iloc[:, :no_of_feature]

    if model_position != None :
        # The Score Position is Selected -> Grab only prefered position and transform
        y = preprocess_score_to_prob(dataset.iloc[:, no_of_feature + model_position-1])
    else :
        y = dataset.iloc[:, no_of_feature:]

    return (np.array(x), np.array(y))

def batch_generator (data_path, batch_size, steps, model_position=None) :
    index = 1
    while True :
        yield load_data_as_batch(data_path, index-1, batch_size, model_position=model_position)
        if index < steps :
            index += 1
        else :
            index = 1

def train_sequencial_model (layers, feature_file, epoch=100, optimiser="adam", loss="categorical_crossentropy", step_per_epoch=20000, model_position=None) :
    data_batch_generator = batch_generator(feature_file, 256, step_per_epoch, model_position=model_position)

    model = Sequential()
    for layer in layers :
        model.add(layer)

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy', 'mse'])    

    training_hist = model.fit_generator(data_batch_generator, epochs=epoch, steps_per_epoch=step_per_epoch, validation_data=data_batch_generator, validation_steps=step_per_epoch)

    return model, training_hist