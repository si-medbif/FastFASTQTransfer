import os
import sys
import time
import glob

## Measuring Compression and Decompression Time Spending by using Generic Compression (gzip)
def is_gzipped (file_name) :
    if file_name.split('.')[-1] == 'gz' :
        return True
    else:
        return False

def unzip_merge_paired_end (base_file_name) :
    os.system("gunzip " + base_file_name + "_1.fastq.gz " + base_file_name + "_2.fastq.gz")
    
    # Merge File
    os.system("cat " + base_file_name + "_1.fastq " + base_file_name + "_2.fastq > " + base_file_name + ".fastq")

def calculate_transfer_time (file_size, speed=1*10**8) :
    # Input : File Size (Byte), Transfer Speed (bps) ; default 100 Mbps
    return (file_size * 8) / speed

def parse_arguments (args) :
    # Add Default Configurations
    if len(args) < 2 :
        exit("Input File is missing")

    result = {
        "input_file" : args[1],
        "output_file": 'output.o',
        "no_of_run" : 10
    }

    for arg in args :
        if arg[:2] == '-n' :
            result['no_of_run'] = arg[3:]
        elif arg[:2] == '-o' :
            result['output_file'] = arg[3:]

    return result

def get_file_name (path_to_file) :
    splited_path = path_to_file.split('/')

    if len(splited_path) == 1 :
        return path_to_file
    else :
        return splited_path[-1]

def prepare_file_list () :
    original_file_list = glob.glob('*.fastq.gz')
    file_list = []
    for file_name in original_file_list :
        file_list.append(file_name.split('_')[0])
    file_list = list(set(file_list))
    return file_list

def run_experiment (file_name, no_of_run, output_file) :
    # Compression and Decompression Experiment
    avg_compress_size, avg_compression_time, avg_decompression_time, avg_transmit_time = 0,0,0,0

    for i in range(no_of_run) :
        # Compression Process
        start_time = time.time()
        os.system('gzip ' + file_name)
        compression_time = time.time() - start_time
        avg_compression_time += compression_time

        # Measured Compressed Size
        compressed_file_size = os.stat(file_name + '.gz').st_size
        avg_compress_size += compressed_file_size

        # Decompression Process
        start_time = time.time()
        os.system('gzip -d ' + file_name + '.gz')
        decompression_time = time.time() - start_time
        avg_decompression_time += decompression_time

        # Get Tranfer Time
        transmit_time = calculate_transfer_time(compressed_file_size)
        avg_transmit_time += transmit_time

        output_file.write(str([i, file_name, os.stat(file_name).st_size, compressed_file_size, transmit_time, compression_time, decompression_time])[1:-1].replace(', ', ',') + '\n')

        return avg_compress_size/no_of_run, avg_compression_time/no_of_run, avg_decompression_time/no_of_run, avg_transmit_time/no_of_run

def main(args) :
    # Get Base File Name
    file_list = prepare_file_list()

    for base_file_name in file_list :
        unzip_merge_paired_end(base_file_name)
        input_file_name = base_file_name + ".fastq"


        print("Run Experiment", args['no_of_run'], ' run(s) on ', input_file_name)

        output_file = open(args['output_file'], 'a')

        # Append Header
        output_file.write("Experiment Number,File Name,Original Size,Compressed Size,Transmitted Time,Compression Time,Decompression Time\n")

        avg_compress_size, avg_compression_time, avg_decompression_time, avg_transmit_time = run_experiment(input_file_name, args['no_of_run'], output_file)

        # Write Final Line with Average
        output_file.write(str([-1, input_file_name, os.stat(input_file_name).st_size, avg_compress_size, avg_transmit_time, avg_compression_time, avg_decompression_time])[1:-1].replace(', ', ',') + "\n")
        output_file.close()

if __name__ == "__main__":
    arguments = parse_arguments(sys.argv)
    main(arguments)