from joblib import Parallel, delayed
import glob
import sys
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def init_position_content (read_length) :
    position_content = {
        "A" : [0] * read_length,
        "T" : [0] * read_length,
        "C" : [0] * read_length,
        "G" : [0] * read_length,
        "N" : [0] * read_length
    }
    return position_content

def transform_position_content_name (position_content) :
    result_position_content = {
        '%A' : position_content['A'],
        '%T' : position_content['T'],
        '%C' : position_content['C'],
        '%G' : position_content['G'],
        '%N' : position_content['N']
    }

    return result_position_content

def transform_for_line_plot (position_content) :
    final_dataframe = {'Position' : [], 'Content Percentages' : [], '%' : []}

    for content in position_content.keys() :
        in_content = position_content[content]

        for i in range(0, len(in_content)) :
            final_dataframe['Position'].append(i)
            final_dataframe['Content Percentages'].append(in_content[i])
            final_dataframe['%'].append(content)

    return pd.DataFrame(final_dataframe)
    
def process_position_content (position_content, line) :
    for i in range(0, len(line)) :
        position_content[line[i]][i] += 1
    
    return position_content

def average_position_content (position_content, no_of_record) :
    for base_content in position_content.keys() :
        for i in range(0,len(position_content[base_content])) :
            position_content[base_content][i] = position_content[base_content][i] * 100 / no_of_record

    return position_content

def plot_line_plot (data, file_name, destination_path) :
    line_plot = sns.barplot(data=data, x='Position', y='Content Percentages', hue='%').set_title("Sequence content across all bases (" + file_name + ")")
    line_plot.figure.set_size_inches(25, 9)
    line_plot.figure.savefig(destination_path + '/' + file_name.split('.')[0] + '_base_content.png', dpi=300)

    plt.close()

def process_single_file (source_path, destination_path) :
    source_file = open(source_path, 'r')
    
    position_content = None
    no_of_record = 0

    line = source_file.readline().replace('\n', '')
    prev = ''

    while line != '' :
        
        if len(prev) > 0 and prev[0] == '@' and line[0] != '@' :
            if position_content == None :
                position_content = init_position_content(len(line))

            position_content = process_position_content(position_content, line)
            no_of_record += 1

        prev = line
        line = source_file.readline().replace('\n', '')

    source_file.close()

    position_content = transform_for_line_plot(transform_position_content_name(average_position_content(position_content, no_of_record)))
    plot_line_plot(position_content, source_path.split('/')[-1], destination_path)

    return position_content

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
    # Single File : python3 plot_base_content.py <FASTQ_file> <Plot Destination Folder>
    # Batch Processing : python3 plot_base_content.py <Source_Folder> <Plot Destination Folder>
    main(sys.argv)