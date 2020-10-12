import os
import sys
import time
import glob
import tracemalloc

## Measuring Compression and Decompression Time Spending by using Generic Compression (gzip, Arithmetic Encoding and Huffman Encoding)
def memory_measurement_closure (command) :
    # Return [Current Memory, Peak Memory]
    tracemalloc.start()
    os.system(command)
    return tracemalloc.get_traced_memory()

def run_experiment_pipeline (compression_command, decompression_command, no_of_run, result_file_path, compressed_file_name, remove_after_finished=False) :
    output_file = open(result_file_path, 'w')
    output_file.write('Experiment No.,Compressed Size (Byte),Compression Time (Sec),Decompression Time (Sec),Compression Memory (MB), Decompression Memory (MB)\n')

    whole_compressed_size, whole_compression_time, whole_decompression_time, whole_compression_memory, whole_decompression_memory = 0,0,0,0,0

    for i in range(no_of_run) :
        # Compression Process
        compression_start_time = time.time()
        compression_peak_memory = memory_measurement_closure(compression_command)[1]
        compression_end_time = time.time()
        compression_time = compression_end_time - compression_start_time

        # Measure Compressed File Size
        compressed_file_size = os.stat(compressed_file_name).st_size
        whole_compressed_size += compressed_file_size

        # Decompression Process
        decompression_start_time = time.time()
        decompression_peak_memory = memory_measurement_closure(decompression_command)[1]
        decompression_end_time = time.time()
        decompression_time = decompression_end_time - decompression_start_time

        whole_compression_time += compression_time
        whole_decompression_time += decompression_time
        whole_compression_memory += (compression_peak_memory / 10**6)
        whole_decompression_memory += (decompression_peak_memory / 10**6)

        # Write the result back to result file
        output_file.write(str(i+1) + ',' + str(compressed_file_size) + ',' + str(compression_time) + ',' + str(decompression_time) + ',' + str(compression_peak_memory) + ',' + str(decompression_peak_memory) + '\n')
    
    output_file.write('-1,' + str(whole_compressed_size/no_of_run) + ',' + str(whole_compression_time/no_of_run) + ',' + str(whole_decompression_time/no_of_run) + ',' + str(whole_compression_memory/no_of_run) + ',' + str(whole_decompression_memory/no_of_run) + '\n')
    output_file.close()

    if remove_after_finished :
        os.system('rm ' + compressed_file_name)

def run_batch_gzip_experiment (file_list, no_of_run, output_file_path) :
    for file_path in file_list :
        run_experiment_pipeline(
            compression_command='gzip -fc ' + file_path + ' > ' + file_path.split('/')[-1] + '.gz',
            decompression_command='gzip -fd ' + file_path.split('/')[-1] + '.gz',
            no_of_run=no_of_run,
            result_file_path=output_file_path + '/gzip_' + file_path.split('.')[0].split('/')[-1] + '.csv',
            compressed_file_name= file_path.split('/')[-1] + '.gz',
            remove_after_finished=True
            )
        os.system('rm ' + file_path.split('/')[-1])

def run_batch_arithmetic_experiment (file_list, no_of_run, output_file_path) :
    for file_name in file_list :
        run_experiment_pipeline(
            compression_command='python3 Experiments/Pilot_Compression/reference-arithmetic-coding/python/arithmetic-compress.py ' + file_name + ' test.bin', 
            decompression_command='python3 Experiments/Pilot_Compression/reference-arithmetic-coding/python/arithmetic-decompress.py test.bin ' + file_name.split('/')[-1], 
            no_of_run = no_of_run, 
            result_file_path = output_file_path + '/arithmetic_' + file_name.split('.')[0].split('/')[-1] + '.csv', 
            compressed_file_name = 'test.bin', 
            remove_after_finished=True )

def run_batch_huffman_experiment (file_list, no_of_run, output_file_path) :
    for file_name in file_list :
        run_experiment_pipeline(
            compression_command='python3 Experiments/Pilot_Compression/Huffman-Coding/Huffman.py -c ' + file_name, 
            decompression_command='python3 Experiments/Pilot_Compression/Huffman-Coding/Huffman.py -d ' + file_name.split('.')[0] + '.huffman', 
            no_of_run = no_of_run, 
            result_file_path = output_file_path + '/huffman_' + file_name.split('.')[0].split('/')[-1] + '.csv', 
            compressed_file_name = file_name.split('.')[0] + '.huffman', 
            remove_after_finished=True )

def main(args) :
    source_folder = args[1]
    result_folder = args[2]
    no_of_run = int(args[3])

    file_list = glob.glob(source_folder + '/*')

    print('There are', len(file_list), 'sample(s) to run.')

    # Run Each Experiment Set
    run_batch_gzip_experiment(file_list, no_of_run, result_folder)
    run_batch_arithmetic_experiment(file_list, no_of_run, result_folder)
    run_batch_huffman_experiment(file_list, no_of_run, result_folder)

if __name__ == "__main__":
    # python3 pilot_compression_test.py <Source_Folder> <Result_Folder> <Number of Run Per Experiment and File>
    main(sys.argv)