from joblib import Parallel, delayed
import glob
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def init_window_stat () :
    window_stat = dict()
    bases = ['A', 'T', 'C', 'G', 'N']

    for base_i in bases :
        for base_j in bases :
            for q_score in range(0,43) :
                window_stat[base_i + base_j + '->' + str(q_score)] = 0

    return window_stat

def update_window_stat (q_score, base, window_stat) :
    for i in range(0,len(base)-2) :
        window_stat[base[i] + base[i+1] + '->' + str(ord(q_score[i+2])-33)] += 1
    
    return window_stat

def convert_score_to_heatmap_representation (window_stat) :
    result_dict = dict()
    
    for key in window_stat.keys() :
        base = key.split('->')[0]
        next_score = int(key.split('->')[1])

        if base not in result_dict :
            result_dict[base] = [0] * 43
        
        result_dict[base][next_score] = window_stat[key]

    return pd.DataFrame(result_dict)

def plot_heatmap (data, file_name, destination_path) :
    heatmap = sns.heatmap(data).set_title("Sliding Windows of Base and Quality Score (" + file_name + ")")
    heatmap.figure.set_size_inches(10, 7)
    heatmap.figure.savefig(destination_path + '/' + file_name.split('.')[0] + '_sliding_windows_qscore_with_base.png', dpi=300)

    plt.close()

def process_single_file (source_path, destination_path) :
    if os.path.exists(destination_path + '/' + source_path.split('/')[-1].split('.')[0] + '_sliding_windows_qscore_with_base.png') :
        print('Plot has been done. Skipping...')
            
    print('Plotting', source_path.split('/')[-1])
    source_file = open(source_path, 'r')
    window_stat = init_window_stat()

    prev = ''
    line = source_file.readline().replace('\n', '')
    current_base = ''
    while line != '' :
        if prev != '' and prev[0] == '@' :
            current_base = line
        elif prev == '+' :
            window_stat = update_window_stat(line, current_base, window_stat)
        prev = line
        line = source_file.readline().replace('\n', '')

    source_file.close()
    
    
    plot_heatmap(convert_score_to_heatmap_representation(window_stat), source_path.split('/')[-1], destination_path)

    return window_stat
    
def main (args) :
    if len(args) > 1 :
        destination_path = args[-1]
    else :
        destination_path = ''

    file_list = glob.glob(args[1] + '/*.fastq')

    print(len(file_list), 'Sample(s) to process.')
    
    if type(file_list) == bool or len(file_list) == 0 :
        process_single_file(args[1], destination_path)
    else:
        Parallel(n_jobs=-1, prefer="processes", verbose=10)(
            delayed(process_single_file)(file_name, destination_path)
            for file_name in file_list
        )

if __name__ == "__main__":
    # Single File : python3 silding_window_score_with_base.py <FASTQ_file> <Plot Destination Folder>
    # Batch Processing : python3 silding_window_score_with_base.py <Source_Folder> <Plot Destination Folder>
    main(sys.argv)