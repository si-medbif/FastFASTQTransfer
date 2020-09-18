from joblib import Parallel, delayed
import glob
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def update_score_line (line, score_line) :
    for score in line :
        score_line[ord(score)-33] += 1
    
    return score_line

def convert_score_to_chart_barchart_representation (score_line) :
    score_list = list(range(1,len(score_line)+1))

    result_dict = {'Score' : score_list, 'Frequency': score_line}
    
    return pd.DataFrame(result_dict)

def plot_bar_chart (data, file_name, img_path) :
    fig = plt.figure(figsize=(27,10), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log', nonposy='clip')
    dist_plot = sns.barplot(data=data, x='Score', y='Frequency').set_title("Score Frequency (" + file_name + ")")
    dist_plot.figure.savefig('Results/Position_Frequency/'+ file_name.split('.')[0] + '_pos_frequency.png')

    plt.close()

def process_single_file (source_path, destination_path) :
    source_file = open(source_path, 'r')
    score_line = [0] * 43

    prev = ''
    line = source_file.readline().replace('\n', '')
    

    while line != '' :
        if prev == '+' :
            score_line = update_score_line(line, score_line)
        prev = line
        line = source_file.readline().replace('\n', '')

    source_file.close()

    plot_bar_chart(convert_score_to_chart_barchart_representation(score_line), source_path.split('/')[-1], destination_path)
    
    return score_line
    
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
    # Single File : python3 plot_score_frequency.py <FASTQ_file> <Plot Destination Folder>
    # Batch Processing : python3 plot_score_frequency.py <Source_Folder> <Plot Destination Folder>
    main(sys.argv)