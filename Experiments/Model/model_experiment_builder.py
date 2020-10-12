import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras import optimizers

# UTILITIES FUNCTION STARTS HERE!

def encode_score (scores) :
    result = []
    for score in scores :
        current_row = [0] * 43
        current_row[score] = 1
        result.append(current_row)
    return result

def transform_LSTM_feature (X) :
    result = list()
    base_dict = {"A" :0,  "T": 1, "C": 2, "G": 3, "N": 4}
    for record in X :
        current_rec_list = list()
        for base in record :
            transformed_base = [0]*len(base_dict)
            transformed_base[base_dict[base]] = 1
            current_rec_list.append(transformed_base)
        result.append(current_rec_list)
    return result

def build_default_feature (X):
    result = list()
    base_dict = {"A" :0,  "T": 1, "C": 2, "G": 3, "N": 4}

    for record in X :
        current_record_result = list()
        for base in record :
            current_base = [0] * len(base_dict)
            current_base[base_dict[base]] = 1
            current_record_result += current_base
        result.append(current_record_result)
    
    return result

def evaluate_model (model, build_hist, X,  Y, Y_transformed, experiment_name) : 
    # print(model.evaluate(X, Y_transformed))

    if experiment_name != 'model_1' :
        print("Accuracy : ", evaluate_accuracy(model.predict(X), Y), ' %')
    
    # Write to Result File
    result_file = open('performance_result.out', 'a')
    result_file.write(experiment_name + "," + str(build_hist.history['loss']) + ',' + str(build_hist.history['categorical_accuracy']) + '\n')
    result_file.close()

    # Dump History to File
    model_hist_file = open('Model_Hist/' + experiment_name + '.model_hist', 'w')
    model_hist_file.write('loss,val_loss,categorical_accuracy,val_categorical_accuracy\n')
    
    for i in range(0,len(build_hist.history['loss'])) :
        model_hist_file.write(str(build_hist.history['loss'][i]) + ',' + str(build_hist.history['val_loss'][i]) + ',' + str(build_hist.history['categorical_accuracy'][i]) + ',' + str(build_hist.history['val_categorical_accuracy'][i]) + '\n')

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
    plt.plot(build_hist.history['categorical_accuracy'])
    plt.plot(build_hist.history['val_categorical_accuracy'])
    plt.plot
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('models/charts/accuracy/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

# Pred_Y as Prob of each Score
def evaluate_accuracy (pred_Y, Y) :
    no_of_positive = 0
    no_of_data = len(pred_Y)

    for record_index in range(0, no_of_data) :
        pred_pos = list(pred_Y[record_index]).index(max(pred_Y[record_index]))
        if pred_pos == Y[record_index] :
            no_of_positive += 1

    print('No of Missed ' , no_of_data - no_of_positive)
    return (no_of_positive/no_of_data) * 100

def evaluate_predicted_model (model_path, X, Y) :
    model = load_model(model_path)

    enc = OneHotEncoder()
    X_transformed = enc.fit_transform(X)
    print("Accuracy : ", evaluate_accuracy(model.predict(X_transformed), Y), ' %')

# MODEL EXPERIMENT STARTS HERE!

def model_1 (X,Y) :
    enc = OneHotEncoder(categories=[range(5)])
    X_transformed = enc.fit_transform(X)

    model = Sequential()
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    hist = model.fit(X_transformed, Y, epochs=10, validation_data=[X_transformed, Y])

    model.save('models/model_1.h5')

    print("Model 1")
    evaluate_model(model, hist, X_transformed, Y, Y, 'model_1')

def model_2 (X,Y, is_encode_feature=False) :

    if is_encode_feature :
        X_transformed = build_default_feature(X)
    else :
        X_transformed = X

    X_transformed = np.array(X_transformed)
    Y_transformed = np.array(encode_score(Y))

    model = Sequential()
    model.add(Dense(43, activation='softmax'))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['categorical_accuracy'])

    hist = model.fit(X_transformed, Y_transformed, epochs=100, validation_data=[X_transformed, Y_transformed])

    # Save Model for later use
    model.save('models/model_2.h5')

    print("Model 2")
    evaluate_model(model, hist, X_transformed, Y, Y_transformed, 'model_2')


def model_3 (X,Y, is_encode_feature=False) :
    if is_encode_feature :
        X_transformed = build_default_feature(X)
    else :
        X_transformed = X

    X_transformed = np.array(X_transformed)
    Y_transformed = np.array(encode_score(Y))

    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(43, activation='softmax'))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['categorical_accuracy'])


    hist = model.fit(X_transformed, Y_transformed, epochs=100, validation_data=[X_transformed, Y_transformed])

    # Save Model for later use
    model.save('models/model_3.h5')

    print("Model 3")
    evaluate_model(model, hist, X_transformed, Y, Y_transformed, 'model_3')

def model_4 (X,Y, is_encode_feature=False) :
    if is_encode_feature :
        X_transformed = build_default_feature(X)
    else :
        X_transformed = X

    X_transformed = np.array(X_transformed)
    Y_transformed = np.array(encode_score(Y))

    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(43, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


    hist = model.fit(X_transformed, Y_transformed, epochs=100, validation_data=[X_transformed, Y_transformed])

    # Save Model for later use
    model.save('models/model_4.h5')

    print("Model 4")
    evaluate_model(model, hist, X_transformed, Y, Y_transformed, 'model_4')

def model_5 (X,Y, is_encode_feature=False) :

    if is_encode_feature :
        X_transformed = build_default_feature(X)
    else :
        X_transformed = X

    X_transformed = np.array(X_transformed)
    Y_transformed = np.array(encode_score(Y))

    model = Sequential()
    model.add(Dense(43, activation='softmax'))

    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    hist = model.fit(X_transformed, Y_transformed, epochs=100, validation_data=[X_transformed, Y_transformed])

    # Save Model for later use
    model.save('models/model_5.h5')

    print("Model 5")
    evaluate_model(model, hist, X_transformed, Y, Y_transformed, 'model_5')

def model_6 (X,Y, is_encode_feature=False) :

    if is_encode_feature :
        X_transformed = build_default_feature(X)
    else :
        X_transformed = X

    X_transformed = np.array(X_transformed)
    Y_transformed = np.array(encode_score(Y))

    model = Sequential()
    model.add(Embedding(500,500))
    model.add(Dense(100, activation="relu"))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(43, activation="softmax"))


    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    hist = model.fit(X_transformed, Y_transformed, epochs=5, validation_data=[X_transformed, Y_transformed])

    # Save Model for later use
    model.save('models/model_6.h5')

    print("Model 6")
    evaluate_model(model, hist, X_transformed, Y, Y_transformed, 'model_6')


# This function will be used to test the model arch by changing the model config
def model_arch_experiment (X,Y, model_name, is_encode_feature=False, layer_configuration=[500]) :
    if is_encode_feature :
        X_transformed = build_default_feature(X)
    else :
        X_transformed = X

    X_transformed = np.array(X_transformed)
    Y_transformed = np.array(encode_score(Y))

    model = Sequential()

    # Hidden Layer with ReLu Function
    for no_of_node in layer_configuration :
        if no_of_node == 0:
            continue

        model.add(Dense(no_of_node, activation='relu'))

    # Output Layer with Softmax function
    model.add(Dense(43, activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    hist = model.fit(X_transformed, Y_transformed, epochs=100, validation_data=[X_transformed, Y_transformed])

    # Save Model for later use
    model.save('models/' + model_name)

    evaluate_model(model, hist, X_transformed, Y, Y_transformed, model_name)

def main (args) :
    # Create the folder for storing result
    os.makedirs('models/charts/accuracy', exist_ok=True)
    os.makedirs('models/charts/loss', exist_ok=True)
    os.makedirs('Model_Hist', exist_ok=True)
    
    # Load Data
    dataset = pd.read_csv(args[1], header=None)

    X = dataset.iloc[:,:505].values
    Y = dataset.iloc[:,505].values
    
    # Filter Score 37 Out
    # dataset = dataset[dataset.iloc[:,100] != 37]

    # Select only Significance Features (19 Features)
    # X = dataset.iloc[:,[500,4,274,499,189,169,164,0,3,404,493,496,480,405,485,490,495,502,503]].values

    # evaluate_predicted_model('models/model_5.h5', X, Y)
    # model_1(X,Y)
    # model_2(X,Y)
    # model_3(X,Y)
    # model_4(X,Y)
    # model_5(X,Y)
    # model_6(X,Y)
    
    for layer_1_node_counter in range(100,500+1,100): 
        for layer_2_node_counter in range(0,500+1, 100) :
            model_arch_experiment (X,Y, 'l1_' + str(layer_1_node_counter) + '_l2_' + str(layer_2_node_counter), is_encode_feature=False, layer_configuration=[layer_1_node_counter, layer_2_node_counter])

if __name__ == "__main__":
    main(sys.argv)
