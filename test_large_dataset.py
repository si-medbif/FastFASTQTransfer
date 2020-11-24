import sys
import math
import matplotlib.pyplot as plt

from Experiments.Model.utilities import generate_training_statistic_file, plot_loss_acc_to_file
from tensorflow.keras.layers import Dense, LSTM
from Experiments.Model.parallel_feature_extraction import parallel_extract_feature, create_fastq_jobs
from sender.model_service import train_sequencial_model, lstm_batch_record_generator, lstm_batch_generator_parallel, SimpleGenerator

def model_training_experiment_set (feature_file_path, model_configuration, layer_configuration, experiment_name, is_lstm=False, model_position=1 ,loss='categorical_crossentropy', generator=None, val_generator=None) :

    print(experiment_name)

    # Train and Evaluate Model
    model, training_hist = train_sequencial_model(layer_configuration, feature_file_path, epoch=model_configuration['no_of_epoch'], step_per_epoch=model_configuration['steps_per_epoch'], model_position=model_position, is_lstm=is_lstm, loss=loss, generator=generator, val_generator=val_generator)    
    
    generate_training_statistic_file(training_hist, experiment_name, destination_file_path='Results/model_experiment/training_stat/' + experiment_name + '.model_hist')
    plot_loss_acc_to_file(training_hist, experiment_name, 'Results/model_experiment/charts/loss', 'Results/model_experiment/charts/accuracy')

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

    # # EXPERIMENT 13 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim']))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_13')

    # # END OF EXPERIMENT 13

    # # EXPERIMENT 14 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_14')

    # # END OF EXPERIMENT 14

    # # EXPERIMENT 15 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_15')

    # # END OF EXPERIMENT 15

    # # EXPERIMENT 16 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_16')

    # # END OF EXPERIMENT 16

    # # EXPERIMENT 17 STARTS HERE

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_17')

    # # END OF EXPERIMENT 17

    # # EXPERIMENT 18 STARTS HERE (From Ex.15 Add Ex.1)

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(50, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_18')

    # # END OF EXPERIMENT 18

    # # EXPERIMENT 19 STARTS HERE (From Ex.15 Add Ex.2)

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_19')

    # # END OF EXPERIMENT 19

    # # EXPERIMENT 20 STARTS HERE (From Ex.15 Add Ex.3)

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(200, activation='softmax'))
    # model_layers.append(Dense(100, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_20')

    # # END OF EXPERIMENT 20

    # # EXPERIMENT 21 STARTS HERE (From Ex.15 Add Ex.4)

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(300, activation='softmax'))
    # model_layers.append(Dense(200, activation='softmax'))
    # model_layers.append(Dense(100, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_21')

    # # END OF EXPERIMENT 21

    # # EXPERIMENT 22 STARTS HERE (From Ex.15 Add Ex.5)

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_22')

    # # END OF EXPERIMENT 22

    # # EXPERIMENT 23 STARTS HERE (From Ex.15 Add Ex.6)

    # # Layer Container
    # model_layers = []

    # # Layer Specification
    # model_layers.append(LSTM(model_configuration['input_dim'], return_sequences=True))
    # model_layers.append(LSTM(90, return_sequences=True))
    # model_layers.append(LSTM(90))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(500, activation='softmax'))
    # model_layers.append(Dense(model_configuration['output_dim'], activation='softmax'))

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration=model_layers, is_lstm=True, experiment_name='Model_23')

    # # END OF EXPERIMENT 23

    # # EXPERIMENT 24 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 43,
    #     'no_of_epoch' : 100,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     LSTM(model_configuration['input_dim'], return_sequences=True),
    #     LSTM(90, return_sequences=True),
    #     LSTM(90),
    #     Dense(500, activation='softmax'),
    #     Dense(500, activation='softmax'),
    #     Dense(500, activation='softmax'),
    #     Dense(500, activation='softmax'),
    #     Dense(500, activation='softmax'),
    #     Dense(model_configuration['output_dim'], activation='softmax')

    # ], is_lstm=True, experiment_name='Model_24')

    # # END OF EXPERIMENT 24

    # # EXPERIMENT 25 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 90,
    #     'no_of_epoch' : 10,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     Dense(model_configuration['input_dim'], activation='relu'),
    #     Dense(50, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ], is_lstm=True, model_position=None, loss='mse', experiment_name='Model_25')

    # # END OF EXPERIMENT 25

    # # EXPERIMENT 26 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 90,
    #     'no_of_epoch' : 10,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     Dense(model_configuration['input_dim'], activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ], is_lstm=True, model_position=None, loss='mse', experiment_name='Model_26')

    # # END OF EXPERIMENT 26

    # # EXPERIMENT 27 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 90,
    #     'no_of_epoch' : 10,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     LSTM(model_configuration['input_dim']),
    #     Dense(200, activation='relu'),
    #     Dense(100, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ], is_lstm=True, model_position=None, loss='mse', experiment_name='Model_27')

    # # END OF EXPERIMENT 27

    # # EXPERIMENT 28 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 90,
    #     'no_of_epoch' : 10,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     LSTM(model_configuration['input_dim']),
    #     Dense(300, activation='relu'),
    #     Dense(200, activation='relu'),
    #     Dense(100, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ], is_lstm=True, model_position=None, loss='mse', experiment_name='Model_28')

    # # END OF EXPERIMENT 28

    # # EXPERIMENT 29 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 90,
    #     'no_of_epoch' : 10,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     LSTM(model_configuration['input_dim']),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ], is_lstm=True, model_position=None, loss='mse', experiment_name='Model_29')

    # # END OF EXPERIMENT 29

    # # EXPERIMENT 30 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 90,
    #     'no_of_epoch' : 10,
    #     'steps_per_epoch': 20000
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     LSTM(model_configuration['input_dim']),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ], is_lstm=True, model_position=None, loss='mse', experiment_name='Model_30')

    # # END OF EXPERIMENT 30

    # # EXPERIMENT 31 STARTS HERE
    # model_configuration = {
    #     'input_dim' : 90,
    #     'output_dim' : 43,
    #     'no_of_epoch' : 100,
    #     'steps_per_epoch': 753677
    # }

    # model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
    #     LSTM(model_configuration['input_dim']),
    #     Dense(500, activation='relu'),
    #     Dense(500, activation='relu'),
    #     Dense(model_configuration['output_dim'], activation='relu')

    # ],
    # generator= lstm_batch_generator_parallel(feature_file_path, n_cpu_core=16, batch_per_core=40000) ,
    # val_generator = lstm_batch_generator_parallel(feature_file_path, n_cpu_core=16, batch_per_core=40000) ,
    # is_lstm=True, experiment_name='Model_31')

    # # END OF EXPERIMENT 31

    # EXPERIMENT 32 STARTS HERE
    no_of_data = 753677
    batch_size = 64
    model_configuration = {
        'input_dim' : 90,
        'output_dim' : 43,
        'no_of_epoch' : 100,
        'steps_per_epoch': math.ceil(no_of_data/batch_size)
    }

    model_training_experiment_set(feature_file_path, model_configuration, layer_configuration = [
        LSTM(model_configuration['input_dim']),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(model_configuration['output_dim'], activation='relu')

    ],
    generator= lstm_batch_generator_parallel(feature_file_path) ,
    val_generator = lstm_batch_generator_parallel(feature_file_path) ,
    is_lstm=True, experiment_name='Model_32')

    # END OF EXPERIMENT 32

if __name__ == "__main__":
    # python test_large_dataset.py <Read File> <Quality Score File> <Feature File Path>
    main(sys.argv)
