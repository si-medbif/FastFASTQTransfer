import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import math

# Plot Base Content Percentage for each position
# INPUT: <Feature File Path> <Plot File Path>
# OUTPUT: <Base Content Percentage for each position in stacked bar plot>

def plot_base_content_percentage_stacked (base_content_df, sample_name, plot_dest) :
    ax = base_content_df.plot.bar(stacked=True, figsize=(20,8))
    plt.title('Base content in percentage (' + sample_name + ')')
    plt.ylabel('Percentage')
    plt.xlabel('Position')
    ax.figure.savefig(plot_dest, dpi=300)

def convert_base_content_to_pandas_df (base_content):
    result_df = {
        'Position' : [i+1 for i in range(0, len(base_content))],
        'A' : [],
        'T' : [],
        'C' : [],
        'G' : [],
        'N' : [],
    }

    for base_content_in_position in base_content :
        for base in base_content_in_position.keys() :
            result_df[base].append(base_content_in_position[base])

    return pd.DataFrame(result_df).set_index('Position')

def convert_base_content_to_percentage (base_content, no_of_read) :
    content_percentage = []

    for base_position in base_content :
        new_base_position = {}

        for key in base_position.keys() :
            new_base_position[key] = base_position[key] * 100 / no_of_read
        
        content_percentage.append(new_base_position)
    
    return content_percentage

def process_line (base_list, base_content) :
    base_converter = {
        '0' : 'N',
        '1' : 'A',
        '2' : 'T',
        '3' : 'C',
        '4' : 'G'
    }

    no_of_position = len(base_list)
    base_list = [base_converter[item] for item in base_list]

    for current_position_number in range(0,no_of_position) :
        base_content[current_position_number][base_list[current_position_number]] += 1
    
    return base_content

def main (args) :
    no_of_read = 0

    sample_name = args[1].split('/')[-1].split('.')[0]
    feature_file = open(args[1], 'r')

    # Get the first line to get read length
    no_of_feature = math.floor(len(feature_file.readline()[:-1].split(','))/2)
    
    # Seek to the starting point
    feature_file.seek(0)
    
    # Init data structure for storing base count in each position
    base_content = []
    for i in range(0, no_of_feature) :
        base_content.append({'A':0, 'T':0, 'C': 0, 'G':0, 'N':0})
  
    for line in feature_file :
        current_line = line[:-1].split(',')[:no_of_feature]

        base_content = process_line(current_line, base_content)
        
        no_of_read += 1

    base_content_percentatge = convert_base_content_to_percentage(base_content, no_of_read)
    base_content_df = convert_base_content_to_pandas_df(base_content_percentatge)
    plot_base_content_percentage_stacked(base_content_df,sample_name, args[2])
        
if __name__ == "__main__":
    main(sys.argv)