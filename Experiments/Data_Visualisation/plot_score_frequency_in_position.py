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

def plot_bar_chart (data, file_name) :
    fig = plt.figure(figsize=(27,10), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log', nonposy='clip')
    dist_plot = sns.barplot(data=data, x='Score', y='Frequency').set_title("Score Frequency (" + file_name + ")")
    dist_plot.figure.savefig(file_name.split('.')[0] + '_pos_frequency.png')

def process_single_file (source_path) :
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

    plot_bar_chart(convert_score_to_chart_barchart_representation(score_line), source_path.split('/')[-1])
    
def main (args) :
    process_single_file(args[1])

if __name__ == "__main__":
    # python3 plot_score_frequency.py <FASTQ_file>
    main(sys.argv)