import sys
import pandas as pd
import numpy as np

from keras.layers import Dense
from sender.parallel_feature_extraction import parallel_extract_feature, create_fastq_jobs
from sender.model_service import train_sequencial_model

def parse_arguments (args) :
    # python3 test_large_dataset.py <Read File> <Quality Score File> <Feature File Path> <Chunk Size>
    return args[1], args[2], args[3], args[4]

def main (args) :
    # create_fastq_jobs(args[1])
    # Producing Feature File From Read and Quality Score File
    read_file_path, qscore_file_path, feature_file_path, chunk_size = parse_arguments(args)
    parallel_extract_feature (read_file_path, qscore_file_path, feature_file_path, chunk_size) 

    # Train the Model
    # input_dim = 100
    # output_dim = 100
    # no_of_epoch = 100
    # steps_per_epoch = 20000
    

    # Layer Specification
    # model_layers = []
    # model_layers.append(Dense(100, activation='relu'))
    # model_layers.append(Dense(200, activation='relu'))
    # model_layers.append(Dense(output_dim, activation='relu'))

    # Load Dataset
    # dataset = pd.read_csv(feature_file_path, header=None)
    # X = dataset.iloc[:, :100]
    # Y = dataset.iloc[:, 100:]

    # model = train_sequencial_model(model_layers, X,Y, X,Y, epoch=no_of_epoch)    
    # model.save('./data/test.model')

if __name__ == "__main__":
    main(sys.argv)
