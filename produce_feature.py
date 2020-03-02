from joblib import Parallel, delayed

import os
import sys
import glob
import random
import math

def produce_string_sep (input_text, sep=',', isEncode=True) :

    if isEncode :
        base_list = encode_base_features(list(input_text))
    else :
        base_list = list(input_text)
    
    return str(base_list).replace("'", "").replace(" ", "")[1:-1]

def encode_base_features (base_string) :
    encoder = {'A' : 0, 'T' : 1, 'C' : 2, 'G' : 3, 'N' : 4}
    feature_series = []

    for base in base_string :
        current_feature = [0] * len(encoder)
        current_feature[encoder[base]] = 1
        feature_series += current_feature
    
    return feature_series
    
def produce_all_feature_set_specific_position (x_file_path, y_file_path, destination_file_path, position_no) :
    x_file = open(x_file_path, 'r')
    y_file = open(y_file_path, 'r')
    destination_file = open(destination_file_path, 'w')

    for x_line in x_file :
        y_line = y_file.readline().strip()
        x_line = x_line.strip()

        current_line = ""

        # 100 Position Per Read
        for base in x_line :
            current_line += base + ','
        
        # Append Answer
        current_line += str(ord(y_line[position_no-1])-33)

        destination_file.write(current_line + "\n")

    x_file.close()
    y_file.close()
    destination_file.close()

def produce_vanilla_feature_by_position (position, base_read_source, base_quality_source, base_destination) :
    read_file = open(base_read_source + "/" + str(position), 'r')
    quality_file = open(base_quality_source + "/" + str(position), 'r')
    output_file = open(base_destination + '/' + str(position), 'w')

    read_line = read_file.readline()
    quality_line = quality_file.readline()

    while read_line != '' and quality_line != '' :
        read_line, quality_line = read_line.strip(), quality_line.strip()
        output_file.write(read_line + ',' + quality_line + '\n')

        read_line = read_file.readline()
        quality_line = quality_file.readline()

    read_file.close()
    quality_file.close()
    output_file.close()

def produce_postion_added_feature (base_read_source, base_quality_source, destination_file, plain_destination_file_path) :
    read_file = open(base_read_source, 'r')
    quality_file = open(base_quality_source, 'r')
    destination_file = open(destination_file, 'w')
    
    # Init File with Header
    destination_file.write('Base,Position,Quality_Score\n')

    # Init Vanilla Header
    vanilla_file = open(plain_destination_file_path, 'w')
    vanilla_file.write('Base,Quality_Score\n')

    read_line = read_file.readline()
    quality_line = quality_file.readline()
    read_counter = 0

    feature_cache = []

    while read_line != '' and read_line != None and quality_line != '' and quality_line != None :
        if len(read_line) < 100 and len(quality_line) < 100 :
            continue

        read_line = read_line.strip()
        quality_line = quality_line.strip()

        # Fixed Length Read 100 Base per Read
        for counter in range(0,100) :
            # Feature -> Base,Position,Quality_Score
            feature_cache.append([read_line[counter], str(counter+1), str(ord(quality_line[counter])-33)])

        read_counter += 1

        if read_counter % 4096 == 0 :
            for feature_item in feature_cache :
                destination_file.write(feature_item[0] + "," + feature_item[1] + "," + feature_item[2] + "\n")
                vanilla_file.write(feature_item[0] + ',' + feature_item[2] + '\n')
            feature_cache = []

        read_line = read_file.readline()
        quality_line = quality_file.readline()

    if len(feature_cache) > 0 :
        for feature_item in feature_cache :
            destination_file.write(feature_item[0] + "," + feature_item[1] + "," + feature_item[2] + "\n")
            vanilla_file.write(feature_item[0] + ',' + feature_item[2] + '\n')

    read_file.close()
    quality_file.close()
    vanilla_file.close()
    destination_file.close()

def produce_dummy_feature (destination_file_path) :
    # Score 0-42
    # (A,T,C,G) -> (33,34,35,36,37)
    # A->0.20, T->0.20, C->0.20, G->0.20, N->0.20
    NO_OF_DATA = 1000000
    BASES = ['A', 'T', 'C', 'G', 'N']
    no_of_each_base = math.ceil(NO_OF_DATA*0.2)
    base_score = {'A' : 38, 'T': 39, 'C':40, 'G':41, 'N':42}

    target_file = open(destination_file_path, 'w')

    for base in base_score :
        for count in range(0,no_of_each_base) :
            score = base_score[base]
            line_string = base

            for base_counter in range(0,99) :
                line_string += ',' + BASES[random.randrange(0,4)]

            line_string += ',' + str(score) + '\n'
            target_file.write(line_string)

    target_file.close()

def produce_vanilla_feature_with_pos_by_position (fastq_file_path, base_destination, position, limit=0) :
    fastq_file = open(fastq_file_path, 'r')
    destination_file = open(base_destination, 'w')

    line_counter = 0
    current_record = []

    for line in fastq_file :
        current_record.append(line.strip())
        line_counter += 1

        if line_counter % 4 == 0 :
            # 0 => Header, 1 => Base, 3 => Quality Score
            base_feature = produce_string_sep(current_record[1])

            # Sample : @CL100097983L1C001R005_196710/2
            split_header = current_record[0].split('_')[1].split('/')
            base_feature += ',' + str(int(current_record[0][3:12])) + ',' + str(int(current_record[0][13])) + ',' + str(int(current_record[0][15:18])) + ',' + str(int(current_record[0][19:22])) + ',' + str(int(split_header[0])) + str(int(split_header[1]))

            q_score_in_pos = str(ord(current_record[3][position-1])-33)

            # Merge to final feature set
            base_feature += "," + q_score_in_pos + "\n"

            # Write Features to file
            destination_file.write(base_feature)

            # Reset feature
            base_feature = ""
            current_record = []
        
        if limit > 0 and line_counter / 4 == limit :
            break

    fastq_file.close()
    destination_file.close()

# This function will produce csv file (read,score,position) by merge feature file from vanilla feature
# Please execute produce_vanilla_feature before use this function otherwise the folder will not be found
# def produce_generalise_feature (vanilla_features_path, destination) :

def main (args) :
    produce_vanilla_feature_with_pos_by_position(args[1], args[2], 1, limit=1000000)

if __name__ == "__main__":
    main(sys.argv)