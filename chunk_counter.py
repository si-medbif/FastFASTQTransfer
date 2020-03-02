from joblib import Parallel, delayed
from PIL import Image

import os
import sys
import glob
import math
import seaborn as sns
import matplotlib.pyplot as plt

def merge_all_dict (raw_dicts) :
    result_dict = raw_dicts[0].copy()

    for current_dict in raw_dicts[1:] :
        for score in current_dict.keys() :
            if score in result_dict :
                result_dict[score] += current_dict[score]
            else :
                result_dict[score] = current_dict[score]

    return result_dict

def write_dict_to_csv(source_dict, destination_file_path, is_convert_to_q_score=False) :
    destination_file = open(destination_file_path, 'w')

    print('Writing to ' + destination_file_path)
    
    for source_key in source_dict.keys() :
        if is_convert_to_q_score :
            destination_file.write(str(ord(source_key)-33) + ',' + str(source_dict[source_key]) + '\n')
        else :
            destination_file.write(str(source_key) + ',' + str(source_dict[source_key]) + '\n')

    destination_file.close()

def process_chunk (source_file_path, destination_folder) :
    source_file = open(source_file_path, 'r')

    os.system('mkdir ' + destination_folder)
    
    # Init Dict
    result_dict_list = []
    for i in range(0,100) :
        result_dict_list.append(dict())

    line = source_file.readline().strip()

    while line != None and line != '' :
        counter = 0

        for score_position in line :
            # Save for each position from 0-99
            if score_position in result_dict_list[counter] :
                result_dict_list[counter][score_position] += 1
            else:
                result_dict_list[counter][score_position] = 1
            counter += 1
        
        line = source_file.readline().strip()

    source_file.close()

    all_position_dict = merge_all_dict(result_dict_list)

    write_dict_to_csv(all_position_dict, destination_folder + '/combined.result' , is_convert_to_q_score=True)

    current_position = 1

    for result_dict in result_dict_list :
        write_dict_to_csv(result_dict, destination_folder + '/' + str(current_position) + '.result', is_convert_to_q_score=True)
        current_position += 1

def merge_chunk_by_position (source_path, position, destination_file) :
    # Read Chunk
    result_dict_list = []
    source_file_list = glob.glob(source_path + '/HS07001_*/' + str(position) + '.result')

    for source_file in source_file_list :
        current_file = open(source_file, 'r')
        current_dict = dict()

        print(source_file)

        line = current_file.readline().strip()

        while line != None and line != '' :
            if len(line.split(',')) != 2 :
                continue
            line = line.split(',')
                
            current_dict[line[0]] = int(line[1])

            line = current_file.readline().strip()

        current_file.close()
        result_dict_list.append(current_dict)

    # Merge Chunk
    all_chunks = merge_all_dict(result_dict_list)

    # Write Chunk to File
    write_dict_to_csv(all_chunks, destination_file)

    print(destination_file + ' has been written.')

def produce_cummulative_plot_by_position (data_dict_base_path, position) :
    # Load CSV to list
    dataset = read_dict_from_file(data_dict_base_path + '/' + str(position) + '.result')

    # Count Record
    record_count = 0
    for item in dataset :
        record_count += int(item[1])
    
    # Calculate Percentage and get only percent
    percentage_record = []
    for item in dataset :
        percentage_record.append(item[1]*100/record_count)

    # Fullill Missing Record with actual count=0
    if len(percentage_record) < 43 :
        record_to_add = 43 - len(percentage_record)    
        for i in range(0, record_to_add) :
            percentage_record.append(0)
    
    percentage_record = sorted(percentage_record)
    percentage_record.reverse()

    current_cum = 0
    cummulative_percentage = []
    # Compute Commulative Percentage
    for item in percentage_record :
        current_cum += item
        cummulative_percentage.append(current_cum)
    
    print(cummulative_percentage)
    # Add x-axis
    x = list(range(0,43))

    # Create Folder for storing Result
    os.system('mkdir commulative_plot')

    fig = plt.figure(figsize=(10,5))
    barplot = sns.barplot(x=x, y=cummulative_percentage).set_title("Position : " + str(position))
    barplot.figure.savefig('commulative_plot/' + str(position) + '.png')



def read_dict_from_file (source_file_path, sep=",", convert_to_int = True) :
    source_file = open(source_file_path, 'r')
    final_list = []

    for line in source_file.readlines() :
        line = line.strip().split(sep)
        if convert_to_int :
            final_list.append([int(line[0]), int(line[1])])
        else :
            final_list.append([line[0], line[1]])

    source_file.close()
    return final_list

def concat_plot (source_path, destination_file) :
    image_files = list()
    
    # Produce File Path
    file_path_list = []
    for i in range(1,101) :
        file_path_list.append(source_path + '/' + str(i) + '.png')
    
    file_path_list.append(source_path + '/combined.png')

    for file_path in file_path_list :
        image_files.append(Image.open(file_path))
    
    images_width, images_height = image_files[0].width, image_files[0].height

    # Find Grid Size
    grid_size = math.ceil(math.sqrt(len(image_files)))
    final_image_file = Image.new('RGB', (images_width*grid_size, images_height*grid_size))
    
    counter = 0
    for i in range(0,grid_size) :
        for j in range(0,grid_size) :

            # The Final Merged Plot might not be perfect square format -> must break when out of images
            if counter > len(image_files)-1 :
                break

            final_image_file.paste(image_files[counter], (j*images_width, i*images_height))
            counter += 1
    final_image_file.save(destination_file)

def main (args) :
    # source_folder = args[1]
    # base_destination_folder = args[2]

    # os.system('mkdir ' + base_destination_folder + '/result')

    # Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    #     delayed(process_chunk)(source_file, base_destination_folder + '/result/' + source_file.split('/')[-1].split('.')[0])
    #     for source_file in glob.glob(source_folder + '/*')
    # )

    # Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    #     delayed(merge_chunk_by_position)(base_destination_folder + '/result', position, 'merged_chunk/' + str(position) + '.result')
    #     for position in range(1,101)
    # )

    # merge_chunk_by_position(base_destination_folder + '/result', 'combined', 'merged_chunk/combined.result')
    
    # Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    #     delayed(produce_cummulative_plot_by_position)('merged_chunk', position)
    #     for position in range(1,101)
    # )

    produce_cummulative_plot_by_position('merged_chunk', 'combined')
    # concat_plot('commulative_plot', './final.png')
if __name__ == "__main__":
    main(sys.argv)    