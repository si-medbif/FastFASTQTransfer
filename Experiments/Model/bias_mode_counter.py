import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot count quality score that equal to MODE of each position
# INPUT: <Feature File Path> <Mode Plot> <Mode Percentage Plot>
# OUTPUT: <Percentage of reads that equal to mode>

def get_mode_from_freq_list (freq_list) :
    max_freq = -1
    max_score = -1

    for score in range(0,43) :
        if freq_list[score] > max_freq :
            max_freq = freq_list[score]
            max_score = score

    return max_score
    
def get_position_mode (frequency_container) :
    result = {
        'Position' : [],
        'Mode' : []
    }

    for position in range(0, len(frequency_container)):
        current_frequency = frequency_container[position]
        result['Mode'].append(get_mode_from_freq_list(current_frequency))
        result['Position'].append(position+1)

    return pd.DataFrame(result)

def get_match_mode_percentage (frequency_container, no_of_read) :
    result = {
        'Position' : [],
        'Matched Percentage': []
    }

    for position in range(0, len(frequency_container)):
        current_frequency = frequency_container[position]
        position_mode = get_mode_from_freq_list(current_frequency)
        result['Position'].append(position+1)
        result['Matched Percentage'].append(current_frequency[position_mode] / no_of_read * 100)

    return pd.DataFrame(result)

def process_line (score_list, frequency_container) :
    position_counter = 0

    for score in score_list :
        frequency_container[position_counter][int(score)] += 1
        position_counter += 1    

    return frequency_container

def main(args) :
    data_file = open(args[1], 'r')
    sample_name = args[1].split('/')[-1].split('.')[0]

    no_of_read = 0

    # Get read length
    line = data_file.readline()[:-1].split(',')
    read_length = math.ceil(len(line) / 2)
    data_file.seek(0)

    # Store score frequency by position
    frequency_container = []
    for i in range(0, read_length) :
        frequency_container.append([0]*43)


    for line in data_file :
        frequency_container = process_line(line[:-1].split(',')[read_length:], frequency_container)
        no_of_read += 1

    data_file.close()

    mode_position = get_position_mode(frequency_container)
    plt.figure(figsize=(30,8))
    bar_plot = sns.barplot(data=mode_position, x='Position', y='Mode')
    plt.xlabel('Position')
    plt.ylabel('Mode')
    plt.title('Mode of quality score in each position (' + sample_name + ')')
    bar_plot.figure.savefig(args[2])
    plt.clf()

    mode_match_percentage = get_match_mode_percentage(frequency_container, no_of_read)
    plt.figure(figsize=(30,8))
    bar_plot = sns.barplot(data=mode_match_percentage ,x='Position', y='Matched Percentage')
    plt.xlabel('Position')
    plt.ylabel('Percentage')
    plt.title('Percentage of quality score matches mode in each position (' + sample_name + ')')
    bar_plot.figure.savefig(args[3])
    plt.clf()

if __name__ == "__main__":
    main(sys.argv)