import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from joblib import Parallel, delayed

def evaluate_model (model, build_hist, X,  Y, experiment_name) : 
    
    # Write to Result File
    result_file = open('performance_result.out', 'a')
    result_file.write(experiment_name + "," + str(build_hist.history['loss']) + ',' + str(build_hist.history['acc']) + '\n')
    result_file.close()

    # Dump History to File
    model_hist_file = open('Model_Hist/' + experiment_name + '.model_hist', 'w')
    model_hist_file.write('loss,val_loss,acc,val_acc\n')
    
    for i in range(0,len(build_hist.history['loss'])) :
        model_hist_file.write(str(build_hist.history['loss'][i]) + ',' + str(build_hist.history['val_loss'][i]) + ',' + str(build_hist.history['acc'][i]) + ',' + str(build_hist.history['val_acc'][i]) + '\n')

    model_hist_file.close()


    # Loss Plot
    plt.plot(build_hist.history['loss'])
    plt.plot(build_hist.history['val_loss'])
    plt.plot
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('models/charts/loss/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

    # Accuracy Plot
    plt.plot(build_hist.history['acc'])
    plt.plot(build_hist.history['val_acc'])
    plt.plot
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('models/charts/accuracy/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

def extract_downsampling_feature (source_file_path) :
    source_data = open(source_file_path, 'r')

    X = []
    Y = []

    for line in source_data :
        # 50% Downsampling

        current_line_feature = extract_downsampling_feature_line(line)
        X.append(current_line_feature[0])
        Y.append(current_line_feature[1])
    
    source_data.close()
    return X,Y

def extract_downsampling_feature_parallel (source_file_path) :
    source_data = open(source_file_path, 'r')

    reads = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(extract_downsampling_feature_line)(line)
        for line in source_data
    )
    source_data.close()

    X = []
    Y = []

    for read in reads :
        X.append(read[0])
        Y.append(read[1])
    
    return X,Y

def extract_downsampling_feature_line (line_data) :
    X = []
    Y = []

    # Precheck if the last char is newline char
    if line_data[-1] == '\n' :
        line_data = line_data[:-1]

    position = 0

    for score in line_data :
        position += 1
        current_score = ord(score)-33

        if position % 2 == 1:
            X.append(current_score)
        else :
            Y.append(current_score)
    
    return [X,Y]

def produce_model (X,Y):

    model = Sequential()
    model.add(Dense(500, input_dim=50, activation='softmax'))
    model.add(Dense(250, activation="softmax"))
    model.add(Dense(125, activation="softmax"))
    model.add(Dense(62, activation="softmax"))
    model.add(Dense(50, activation="softmax"))

    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    hist = model.fit(X, Y, epochs=100, validation_data=[X, Y])

    evaluate_model(model, hist, X, Y, 'SS')

    return model, hist


def main (args):
    quality_file_path = args[1]

    X,Y = extract_downsampling_feature_parallel(quality_file_path)
    X = np.array(X)
    Y = np.array(Y)

    print("Downsampling Feature Done!")

    model,hist = produce_model(X,Y)

if __name__ == "__main__":
    main(sys.argv)