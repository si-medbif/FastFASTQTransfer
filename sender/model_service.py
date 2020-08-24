import pandas as pd
import numpy as np

from keras.models import Sequential 

# Buold and Fit the model

def load_data_as_batch (data_path, index, batch_size) :
    dataset = pd.read_csv(data_path, skiprows=index*batch_size, nrows=batch_size)

    no_of_feature = len(dataset.columns) / 2

    # Seperate X and Y by Half
    x = dataset.iloc[:, :no_of_feature]
    y = dataset.iloc[:, no_of_feature+1:]

    return (np.array(x), np.array(y))

def batch_generator (data_path, batch_size, steps) :
    index = 1
    while True :
        yield load_data_as_batch(data_path, index-1, batch_size)
        if index < steps :
            index += 1
        else :
            index = 1

def train_sequencial_model (layers, X, Y, val_X, val_Y, epoch=100, optimiser="sgd", loss="categorical_crossentropy") :
    model = Sequential()
    for layer in layers :
        model.add(layer)

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])    

    # model.fit_generator(data_batch_generator, epochs=no_of_epoch, steps_per_epoch=steps_per_epoch, validation_data=data_batch_generator, validation_steps=steps_per_epoch, use_multiprocessing=True)
    # data_batch_generator = batch_generator(feature_file_path, 256, steps_per_epoch)

    model.fit(X,Y, epoch=epoch, validation_data=(X,Y))

    return model