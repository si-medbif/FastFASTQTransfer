import sys
import pandas as pd

# Extract pilot compression result from Tool/Pilot_Compression
# INPUT: <RESULT_FILE> <SW NAME> <DESTINATION FILE>

def main (args) :
    input_file = open(args[1], 'r')
    sw_name = args[2]
    destination_file = open(args[3], 'w')

    # Write Header to the destination
    destination_file.write('c_real_time,c_kernel_time,c_user_time,c_compute_percentage,c_avg_memory,c_max_memory,compressed_size,d_real_time,d_kernel_time,d_user_time,d_compute_percentage,d_avg_memory,d_max_memory,sample_name,sw_name\n')

    file_name = ''
    line = input_file.readline()
    while line != '' :

        # File Name Header
        if line[:-1].split('.')[-1] == 'quality' or line[:3] == 'ERR' :
            file_name = line[:-1].split('/')[-1].split('.')[0]
            line = input_file.readline()
            continue

        compression_line = line[:-1]
        compressed_size = input_file.readline()[:-1]
        decompression_line = input_file.readline()[:-1]

        final_line = compression_line + ',' + compressed_size + ',' + decompression_line + ',' + file_name + ',' + sw_name + '\n'
        destination_file.write(final_line)
        line = input_file.readline()

    input_file.close()
    destination_file.close()

if __name__ == "__main__":
    main(sys.argv)