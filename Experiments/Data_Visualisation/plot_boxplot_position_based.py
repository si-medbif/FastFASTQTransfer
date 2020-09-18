from joblib import Parallel, delayed

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import os
import multiprocessing as mp
import math

def process_position_data (position_data, line) :    
    for i in range(0, len(line)) :
        raw_score = ord(line[i]) - 33
        position_data['Position within Read'].append(i+1)
        position_data['Quality Score'].append(raw_score)

    return position_data

def plot_boxplot (data, file_name, destination_path) :
    boxplot = sns.boxplot(data = pd.DataFrame(data), x='Position within Read', y='Quality Score').set_title("Quality Score Across All Bases (" + file_name + ")")
    boxplot.figure.set_size_inches(25, 9)
    boxplot.figure.savefig(destination_path + '/' + file_name.split('.')[0] + '_score_boxplot.png', dpi=300)
    plt.close()

def process_single_file (input_path, destination_path) :
    print('Plotting', input_path.split('/')[-1])

    input_file = open(input_path, 'r')
    line = input_file.readline().replace('\n', '')
    prev = ''
    position_data = {'Position within Read' : [], 'Quality Score' : []}

    while line != '' :
        if prev == '+' :

            position_data = process_position_data(position_data, line)

        prev = line
        line = input_file.readline().replace('\n', '')

    input_file.close()

    plot_boxplot(position_data, input_path.split('/')[-1], destination_path)
    return position_data

def main(args) :
    source_path = args[1]
    
    if len(args) > 2 :
        destination_path = args[2]
    else :
        destination_path = ''

    file_list = glob.glob(source_path + '/*.fastq')
    

    if type(file_list) == bool or len(file_list) == 0 :
        process_single_file(source_path, destination_path)
    else :
        print(len(file_list), 'Sample(s) to process.')
        Parallel(n_jobs=1, prefer="processes", verbose=10)(
            delayed(process_single_file)(file_name, destination_path)
            for file_name in file_list
        )

if __name__ == "__main__":
    # Single File : python3 plot_boxplot_position_based.py <FASTQ_file> <Plot Destination Folder>
    # Batch Processing : python3 plot_boxplot_position_based.py <Source_Folder> <Plot Destination Folder>
    main(sys.argv)