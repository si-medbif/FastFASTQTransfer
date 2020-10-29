import sys
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, LSTM
from Experiments.Model.parallel_feature_extraction import parallel_extract_feature, create_fastq_jobs
from sender.model_service import train_sequencial_model

def evaluate_model (training_history, experiment_name) : 

    # Dump History to File
    model_hist_file = open('Results/model_experiment/training_stat/' + experiment_name + '.model_hist', 'w')
    model_hist_file.write('loss,val_loss,acc,val_acc\n')
    
    for i in range(0,len(training_history.history['loss'])) :
        model_hist_file.write(str(training_history.history['loss'][i]) + ',' + str(training_history.history['val_loss'][i]) + ',' + str(training_history.history['accuracy'][i]) + ',' + str(training_history.history['val_accuracy'][i]) + '\n')

    model_hist_file.close()


    # Loss Plot
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.plot
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('Results/model_experiment/charts/loss/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

    # Accuracy Plot
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.plot
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('Results/model_experiment/charts/accuracy/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

def model_training_experiment_set (feature_file_path, model_configuration, layer_configuration, experiment_name, is_lstm=False) :

    # Train and Evaluate Model
    model, training_hist = train_sequencial_model(layer_configuration, feature_file_path, epoch=model_configuration['no_of_epoch'], step_per_epoch=model_configuration['steps_per_epoch'], model_position=1, is_lstm=is_lstm)    
    evaluate_model(training_hist, experiment_name)
    
    # Save trained model to file
    model.save('Results/model_experiment/model/' + experiment_name + '.h5')

def main (args) :    
    # Input Parameter
    read_file_path, qscore_file_path, feature_file_path = args[1], args[2], args[3]
    
    # Producing Feature File From Read and Quality Score File
    # parallel_extract_feature (read_file_path, qscore_file_path, feature_file_path)

    # Training Configurations
    model_configuration = {
        'input_dim' : 90,
        'output_dim' : 43,
        'no_of_epoch' : 10,
        'steps_per_epoch': 200000
    }

    # # EXPERIMENT 1 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='softmax'))
    # model_layers.append(Dense(50, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='3_Layers_Softmax_50L2')

    # # END OF EXPERIMENT 1

    # # EXPERIMENT 2 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='3_Layers_Softmax_500L2')

    # # END OF EXPERIMENT 2

    # # EXPERIMENT 3 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='softmax'))
    # model_layers.append(Dense(200, activation='softmax'))
    # model_layers.append(Dense(100, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='4_Layers_Softmax_200L2_100L3')

    # # END OF EXPERIMENT 3

    # # EXPERIMENT 4 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='softmax'))
    # model_layers.append(Dense(300, activation='softmax'))
    # model_layers.append(Dense(200, activation='softmax'))
    # model_layers.append(Dense(100, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_4')

    # # END OF EXPERIMENT 4

    # # EXPERIMENT 5 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_5')

    # # END OF EXPERIMENT 5

    # # EXPERIMENT 6 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='4_Layers_Softmax_200L2_100L3')

    # # END OF EXPERIMENT 6

    # # EXPERIMENT 7 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='relu'))
    # model_layers.append(Dense(50, activation='relu'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_7')

    # # END OF EXPERIMENT 7

    # # EXPERIMENT 8 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_8')

    # # END OF EXPERIMENT 8

    # # EXPERIMENT 9 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='relu'))
    # model_layers.append(Dense(200, activation='relu'))
    # model_layers.append(Dense(100, activation='relu'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_9')

    # # END OF EXPERIMENT 9

    # # EXPERIMENT 10 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='relu'))
    # model_layers.append(Dense(300, activation='relu'))
    # model_layers.append(Dense(200, activation='relu'))
    # model_layers.append(Dense(100, activation='relu'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_10')

    # # END OF EXPERIMENT 10

    # # EXPERIMENT 11 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_11')

    # # END OF EXPERIMENT 11

    # # EXPERIMENT 12 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(Dense(model_configuration['input_dim'], activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(500, activation='relu'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, experiment_name='Model_12')

    # # END OF EXPERIMENT 12

    # EXPERIMENT 13 STARTS HERE

    # Layer Container
    model_layers = []

    # Layer Specification
    model_layers.append(LSTM(model_configuration['input_dim']))
    model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_13')

    # END OF EXPERIMENT 13

if __name__ == "__main__":
    # python test_large_dataset.py <Read File> <Quality Score File> <Feature File Path>
    main(sys.argv)
