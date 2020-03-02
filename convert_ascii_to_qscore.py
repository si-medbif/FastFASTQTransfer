from joblib import Parallel, delayed
import sys
import os
import glob
import random

def transform_file (file_path, destination) :
    input_file = open(file_path, 'r')

    full_destination_path = destination + '/' + file_path.split('\\')[-1]

 
    line_counter = 1
    lines_buffer = []
    score_line = input_file.readline()

    while score_line != "" :       
        lines_buffer.append(str(int(score_line.strip())-33))

        if line_counter == 10000000:
            write_to_file(lines_buffer, full_destination_path)
            lines_buffer.clear()
            line_counter = 0

        line_counter += 1
        score_line = input_file.readline()

    input_file.close()

def write_to_file (lines, destination) :
    output_file = open(destination, 'a')

    for line in lines :
        output_file.write(line + '\n')
    output_file.close()

def main (args) :
    Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(transform_file)(file_name, 'position_process/quality_merged_transformed')
        for file_name in sorted(glob.glob('position_process/quality_merged/*'))
    )

if __name__ == "__main__":
    main(sys.argv)