from joblib import Parallel, delayed
import glob
import sys
import pandas as pd
import seaborn as sns
import pandas as pd
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

def plot_heatmap (data, file_name) :
    heatmap = sns.heatmap(data).set_title("Sliding Windows of Base and Quality Score (" + file_name + ")")
    heatmap.figure.set_size_inches(10, 7)
    heatmap.figure.savefig(file_name.split('.')[0] + '_sliding_windows_qscore_with_base.png', dpi=300)

def process_single_file (source_path) :
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
    
    
    plot_heatmap(convert_score_to_heatmap_representation(window_stat), source_path.split('/')[-1])

def main (args) :
    process_single_file(args[1])

if __name__ == "__main__":
    # python3 silding_window_score_with_base.py <FASTQ_file>
    main(sys.argv)