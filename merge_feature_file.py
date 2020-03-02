import glob
import shutil
import math
import sys
import random

def merge_feature_complete_file (source_folder, destination_file, is_random=False) :
    complete_file_list = glob.glob(source_folder + '/*.feature_chunk')

    if is_random :
        random.shuffle(complete_file_list)

    # Copy the first file to the destination file
    shutil.copy(complete_file_list[0], destination_file)

    complete_file_list = complete_file_list[1:]

    destination_file = open(destination_file, 'a')

    for file_name in complete_file_list :
        print('Processing ' + file_name.split('/')[-1])

        current_file = open(file_name, 'r')

        current_line = current_file.readline()

        while current_line != None and current_line != '' :
            destination_file.write(current_line.strip() + '\n')
            current_line = current_file.readline()

        current_file.close()

    destination_file.close()

def split_test_train_to_file (source_feature_file, training_ratio, training_desitnation, testing_destination) :
    
    # Get no of record
    source_file = open(source_feature_file, 'r')
    
    line_counter = 0

    for line in source_file :
        line_counter += 1

    source_file.close()


    no_of_training_rec = math.ceil(line_counter * training_ratio)
    no_of_testing_rec = line_counter - no_of_training_rec

    line_counter = 0

    training_file = open(training_desitnation, 'w')
    testing_file = open(testing_destination, 'w')

    for line in source_file :
        line_counter += 1
        if line_counter <= no_of_training_rec :
            training_file.write(line.strip() + '\n')
        else :
            testing_file.write(line.strip() + '\n')
    
    training_file.close()
    testing_file.close()

    return no_of_training_rec, no_of_testing_rec

def main (args) :
    # merge_feature_complete_file(args[1], args[2])
    no_of_training, no_of_tesing = split_test_train_to_file(args[2], float(args[3]), args[4], args[5])

    print('No of Training : ', no_of_training, 'No of Testing : ', no_of_tesing)

if __name__ == "__main__":
     # python merge_feature_file.py <Source Folder> <Merged Destination Folder> <Training Raito> <Training Destination> <Testing Desitnation>
    main(sys.argv)