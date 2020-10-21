import sys
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
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

def main (args) :    
    # Input Parameter
    read_file_path, qscore_file_path, feature_file_path, chunk_size = args[1], args[2], args[3], args[4]
    
    # Producing Feature File From Read and Quality Score File
    parallel_extract_feature (read_file_path, qscore_file_path, feature_file_path, chunk_size) 
    
    # Training Configurations
    input_dim = 90
    output_dim = 43
    no_of_epoch = 1
    steps_per_epoch = 20000

    # Layer Container
    model_layers = []

    # Layer Specification
    model_layers.append(Dense(input_dim, activation='relu'))
    model_layers.append(Dense(200, activation='relu'))
    model_layers.append(Dense(output_dim, activation='softmax'))

    # Train and Evaluate Model
    model, training_hist = train_sequencial_model(model_layers, feature_file_path, epoch=no_of_epoch, step_per_epoch=steps_per_epoch, model_position=1)    
    evaluate_model(training_hist, '3_Layers_Softmax')

if __name__ == "__main__":
    # python test_large_dataset.py <Read File> <Quality Score File> <Feature File Path> <Chunk Size>
    main(sys.argv)
