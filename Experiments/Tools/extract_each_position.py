from joblib import Parallel, delayed

import os
import sys
import glob

def chunk_process_quality_score (chunk_file_name) :
    chunk_process_distribute(chunk_file_name, 'position_process/quality/position_based', is_quality_score=True)

def chunk_process_read (chunk_file_name) :
    chunk_process_distribute(chunk_file_name, 'position_process/read/position_based')

def chunk_process_distribute(chunk_file_path, destination, is_quality_score=False) :
    chunk_file = open(chunk_file_path, 'r')

    read = chunk_file.readline()
    
    # Init New Chunk
    chunk_distribution_read = dict()

    chunk_file_name = chunk_file_path.split('/')[-1]
    os.system('mkdir ' +  destination + '/' + chunk_file_name)
    
    read_counter = 0

    while read != "":
        position_counter= 1
        
        for base in read :
            if base == '\n' :
                continue
                
            if position_counter not in chunk_distribution_read :
                chunk_distribution_read[position_counter] = list()
            
            if is_quality_score is True :
                chunk_distribution_read[position_counter].append(ord(base)-33)
            else :
                chunk_distribution_read[position_counter].append(base)            
        
            position_counter += 1
        
        if read_counter % 1000000 == 0:
            write_dictionary_to_specific_file(chunk_distribution_read, destination + '/' + chunk_file_name)         
            chunk_distribution_read.clear()
        
        read = chunk_file.readline()
        read_counter += 1
    
    # Write each dictionary to specific file
    write_dictionary_to_specific_file(chunk_distribution_read, destination + '/' + chunk_file_name) 
    chunk_distribution_read.clear()

    chunk_file.close()

def write_dictionary_to_specific_file (input_dictionary, destination) :
        for dict_key in input_dictionary.keys() :
            current_file = open(destination  + '/' + str(dict_key), 'a')

            for item in input_dictionary[dict_key] :
                current_file.write(str(item) + '\n')
            
            current_file.close()

def merge_quality_chunk () :
    root_result_folder = 'position_process/quality_merged'
    chunk_folder = 'position_process/quality/position_based/quality_*'

    os.system('mkdir ' + root_result_folder)

    Parallel(n_jobs=10, prefer="processes", verbose=6)(
        delayed(merge_by_position)(position, chunk_folder, root_result_folder)
        for position in range(63, 101)
    )

def merge_read_chunk () :
    root_result_folder = 'position_process/read_merge'
    chunk_folder = 'position_process/read/position_based/read_*'

    os.system('mkdir ' + root_result_folder)
    
    Parallel(n_jobs=10, prefer="processes", verbose=6)(
        delayed(merge_by_position)(position, chunk_folder, root_result_folder)
        for position in range(1, 101)
    )

def merge_by_position (position, chunk_folder, result_folder) :
    result_file = result_folder + '/' + str(position)

    print("Merging Position " + str(position) + " of 100")
    os.system('cat ' + chunk_folder + "/" + str(position) + ' > ' + result_file)

def main (args) :
    # Split the file
    os.system('rm -rf position_process')
    os.system('mkdir position_process')
    os.system('mkdir position_process/quality && mkdir position_process/read')
    os.system('split -l10000000 ' + args[2] + ' position_process/quality/quality_)
    os.system('split -l10000000 ' + args[1] + ' position_process/read/read_')

    # Delete Last Run Result
    os.system('rm -rf position_process/quality/position_based')
    os.system('mkdir position_process/quality/position_based')

    os.system('rm -rf position_process/read/position_based')
    os.system('mkdir position_process/read/position_based') 

    # Execute position runner
    Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(chunk_process_quality_score)(file_name)
        for file_name in glob.glob('position_process/quality/quality_*')
    )

    Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(chunk_process_read)(file_name)
        for file_name in glob.glob('position_process/read/read_*')
    )

    # Remove old Result
    os.system('rm -rf position_distribution/quality_merged')
    os.system('rm -rf position_distribution/read_merged')

    # Merge the result back
    merge_quality_chunk()
    merge_read_chunk()

    # Remove Unmerged Result
    os.system('rm -rf position_distribution/quality')
    os.system('rm -rf position_distribution/read')


if __name__ == "__main__":
    main(sys.argv)