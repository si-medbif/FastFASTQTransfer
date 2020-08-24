from joblib import Parallel, delayed

import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import os
import multiprocessing as mp
import math

def count_line_from_file (file_full_path) :
    input_file = open(file_full_path, 'r')

    line_counter = 0
    line_content = input_file.readline()

    while line_content != "" :
        line_counter += 1
        input_file.readline()

    input_file.close()

    return line_counter

def produce_frequency_frame (file_full_path) :
    input_file = open(file_full_path, 'r')
    
    frequency_dict = dict()
    current_line = input_file.readline()

    while current_line != '' :
        current_score = int(current_line.strip())

        if current_score not in frequency_dict :
            frequency_dict[current_score] = 1
        else :
            frequency_dict[current_score] += 1

        current_line = input_file.readline()
    
    # Save Dict to Persistance Storage for futher process
    persistance_file = open('position_process/quality_plot_dict/' + file_full_path.split('/')[-1], 'ab')
    pickle.dump(transform_frequency_dict_order(frequency_dict), persistance_file)
    persistance_file.close()

    input_file.close()
    return frequency_dict

def transform_frequency_dict_order (input_dict) :
    # Illumina Quality Score ranking from 0 to 40
    for i in range(0,41) :
        if i not in input_dict :
            input_dict[i] = 0

    return input_dict
def reduce_file_to_frequency (file_full_path) :
    freq_frame = produce_frequency_frame(file_full_path)
    return pd.DataFrame(data={'quality_score' : list(freq_frame.keys()), 'frequency' : list(freq_frame.values())})

def load_dictionary_from_file (file_full_path) :
    input_dict = open(file_full_path, 'rb')
    dataset = transform_frequency_dict_order(pickle.load(input_dict))
    input_dict.close()
    return pd.DataFrame(data={'quality_score' : list(dataset.keys()), 'frequency' : list(dataset.values())})

def generate_dist_plot (source_file_name, destination) :
    # dataset = reduce_file_to_frequency(source_file_name)
    dataset = load_dictionary_from_file(source_file_name)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log', nonposy='clip')

    dist_plot = sns.barplot(data=dataset, x='quality_score', y='frequency').set_title("Position : " + source_file_name.split('/')[-1]) 
    dist_plot.figure.savefig(destination + '/' + source_file_name.split('/')[-1] + '.png')

def main(args) :
    source_path = args[1]
    destination_path = args[2]
    
    file_list = glob.glob(source_path + '/*')

    os.system('mkdir ' + destination_path)

    Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(generate_dist_plot)(file_name, destination_path)
        for file_name in file_list
    )

if __name__ == "__main__":
    # RUN : python3 plot_position_based.py <source_path> <destination_path>
    main(sys.argv)