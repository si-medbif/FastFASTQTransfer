from joblib import Parallel, delayed
import glob
import sys
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def init_window_stat () :
    window_stat = dict()

    for i in range(0,43) :
        for j in range(0,43) :
            for k in range(0,43) :
                window_stat[str(i) + ',' + str(j) + '->' + str(k)] = 0

    return window_stat

def update_window_stat (line, window_stat) :
    for i in range(0,len(line)-2) :
        window_stat[str(ord(line[i])- 33) + ',' + str(ord(line[i+1])- 33) + '->' + str(ord(line[i+2])- 33)] += 1
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
    heatmap = sns.heatmap(data).set_title("Sliding Windows of Base (" + file_name + ")")
    heatmap.figure.set_size_inches(10, 7)
    heatmap.figure.savefig(destination_path + '/' + file_name.split('.')[0] + '_sliding_windows_qscore.png', dpi=300)

    plt.close()

def process_single_file (source_path, destination_path) :
    source_file = open(source_path, 'r')
    window_stat = init_window_stat()

    prev = ''
    line = source_file.readline().replace('\n', '')

    while line != '' :
        if prev == '+' :
            window_stat = update_window_stat(line, window_stat)
        prev = line
        line = source_file.readline().replace('\n', '')

    source_file.close()
    
    plot_heatmap(convert_score_to_heatmap_representation(window_stat), source_path.split('/')[-1], destination_path)

def main (args) :
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
        Parallel(n_jobs=-1, prefer="processes", verbose=10)(
            delayed(process_single_file)(file_name, destination_path)
            for file_name in file_list
        )

if __name__ == "__main__":
    # python3 silding_window_q_score.py <FASTQ_file>
    main(sys.argv)