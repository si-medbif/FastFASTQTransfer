from joblib import Parallel, delayed
import glob
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def process_q_score_line (q_score_line) :
    current_score_line = []

    for score in q_score_line :
        current_score_line.append(ord(score) - 33)

    return current_score_line

def merge_q_score_line (original_score_line, new_score_line) :
    result_score_line = []

    for i in range(0, min(len(original_score_line), len(new_score_line))) :
        result_score_line.append(original_score_line[i] + new_score_line[i])

    return result_score_line

def avg_score_line (q_score_line, no_of_record) :
    result_score_line = []
    for score in q_score_line :
        result_score_line.append(score/no_of_record)

    return result_score_line

def convert_score_to_chart_barchart_representation (score_line) :
    order_list = list(range(1,len(score_line)+1))

    result_dict = {'Position' : order_list, 'Average Score': score_line}
    
    return pd.DataFrame(result_dict)

def plot_bar_chart (data, file_name) :
    fig = plt.figure(figsize=(27,10), dpi=300)
    dist_plot = sns.barplot(data=data, x='Position', y='Average Score').set_title("Average Score in Position (" + file_name + ")")
    dist_plot.figure.savefig(file_name.split('.')[0] + '_pos_avg.png')

def process_single_file (source_file) :
    score_line = list()
    no_of_record = 0

    input_file = open(source_file, 'r')

    prev_line = ''
    line = input_file.readline().replace('\n', '')

    while line != '' :
        if prev_line == '+' :
            if score_line == [] :
                score_line = process_q_score_line(line)
            else :
                score_line = merge_q_score_line(score_line, process_q_score_line(line))
            no_of_record += 1

        prev_line = line
        line = input_file.readline().replace('\n', '')
    
    input_file.close()

    score_line = avg_score_line(score_line, no_of_record)
    plot_bar_chart(convert_score_to_chart_barchart_representation(score_line), source_file.split('/')[-1])

def main (args) :
    source_folder = args[1]
    process_single_file(source_folder)
    Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(process_single_file)(file_name)
        for file_name in glob.glob(source_folder + '/*.fastq')
    )

if __name__ == "__main__":
    # python3 plot_avg_score_position.py <FASTQ_file>
    main(sys.argv)