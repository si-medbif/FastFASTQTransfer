import sys
import glob
import math

# Extracting the result from pilot compression bash script
def convert_time_to_sec (time_string) :
    time_components = time_string.split('.')[0].split(':')
    return str(int(time_components[0]) * 60 + int(time_components[1]))

def extract_arithmetic_result (file_size_result_file_path, time_result_file_path, transform_result_path) :
    result = {}

    file_size_result_file = open(file_size_result_file_path, 'r')
    line = file_size_result_file.readline()

    while line != '' :
        elements = line.split(' ')
        file_size = int(elements[5]) / 10**9
        sample_name = elements[-1].split('.')[0]

        result[sample_name] = {
            'compressed_size' : file_size
        }
        
        line = file_size_result_file.readline()

    file_size_result_file.close()

    
    time_result_file = open(time_result_file_path, 'r')
    line = time_result_file.readline()
    line_flag = 0

    while True :
        if line == '' :
            break
        
        elements = line[:-1].split(',')
        if (len(elements) != 4) :
            sample_name = line.replace('\n', '')
            result[sample_name]['compression'] = {
                'real_time' : [],
                'cpu_time' : [],
                'avg_memory_used' : [],
                'peak_memory_used' : []
            }
            result[sample_name]['decompression'] = {
                'real_time' : [],
                'cpu_time' : [],
                'avg_memory_used' : [],
                'peak_memory_used' : []
            }
            line_flag = 0

        else:
            if line_flag == 0:
                # Elapsed Real Time, Total CPU Time, Avg Memory Used (KB), Peak Memory Used (KB)
                result[sample_name]['compression']['real_time'].append(elements[0])
                result[sample_name]['compression']['cpu_time'].append(elements[1])
                result[sample_name]['compression']['avg_memory_used'].append(elements[2])
                result[sample_name]['compression']['peak_memory_used'].append(elements[3])
                
                line_flag = 1
            else :
                result[sample_name]['decompression']['real_time'].append(elements[0])
                result[sample_name]['decompression']['cpu_time'].append(elements[1])
                result[sample_name]['decompression']['avg_memory_used'].append(elements[2])
                result[sample_name]['decompression']['peak_memory_used'].append(elements[3])

                line_flag = 0
        line = time_result_file.readline()
    time_result_file.close()

    # Transform Result
    transformed_result_file = open(transform_result_path, 'w')

    # Write Header To File
    transformed_result_file.write('sample_name,compressed_size,c_real_time,c_total_cpu_time,c_avg_mem,c_peak_mem,d_real_time,d_total_cpu_time,d_avg_mem,d_peak_mem\n')

    for sample_name in result.keys() :
        compressed_size = result[sample_name]['compressed_size']
        no_of_experiment = len(result[sample_name]['compression']['real_time'])

        for i in range(0, no_of_experiment) :
            transformed_result_file.write(sample_name + ',' + str(compressed_size) + ',' + result[sample_name]['compression']['real_time'][i] + ',' + result[sample_name]['compression']['cpu_time'][i] + ',' + result[sample_name]['compression']['avg_memory_used'][i] + ',' + result[sample_name]['compression']['peak_memory_used'][i] + ',' + result[sample_name]['decompression']['real_time'][i] + ',' + result[sample_name]['decompression']['cpu_time'][i] + ',' + result[sample_name]['decompression']['avg_memory_used'][i] + ',' + result[sample_name]['decompression']['peak_memory_used'][i] + '\n')

    transformed_result_file.close()

def extract_huffman_result (result_path, transform_result_path) :
    result_file = open(result_path , 'r')
    result = {}
    
    line = result_file.readline()
    while line != '' :
        line = line[:-1]

        if len(line.split(',')) != 4 :
            sample_name = line
            result[sample_name] = {}
            result[sample_name]['compression'] = {
                'real_time' : [],
                'cpu_time' : [],
                'avg_memory_used' : [],
                'peak_memory_used' : []
            }
            result[sample_name]['decompression'] = {
                'real_time' : [],
                'cpu_time' : [],
                'avg_memory_used' : [],
                'peak_memory_used' : []
            }
        else :
            compression_stat = line.split(',')
            decompression_stat = result_file.readline()[:-1].split(',')
            
            time_items = compression_stat[0].split('.')[0].split(':')
            result[sample_name]['compression']['real_time'].append(convert_time_to_sec(compression_stat[0]))
            result[sample_name]['compression']['cpu_time'].append(compression_stat[1])
            result[sample_name]['compression']['avg_memory_used'].append(compression_stat[2])
            result[sample_name]['compression']['peak_memory_used'].append(compression_stat[3])

            if len(decompression_stat) == 4 :
                result[sample_name]['decompression']['real_time'].append(convert_time_to_sec(decompression_stat[0]))
                result[sample_name]['decompression']['cpu_time'].append(decompression_stat[1])
                result[sample_name]['decompression']['avg_memory_used'].append(decompression_stat[2])
                result[sample_name]['decompression']['peak_memory_used'].append(decompression_stat[3])

        line = result_file.readline()

    result_file.close()

    # Transform Result
    transformed_result_file = open(transform_result_path, 'w')

    # Write Header To File
    transformed_result_file.write('sample_name,c_real_time,c_total_cpu_time,c_avg_mem,c_peak_mem,d_real_time,d_total_cpu_time,d_avg_mem,d_peak_mem\n')

    for sample_name in result.keys() :
        no_of_experiment = min(len(result[sample_name]['compression']['real_time']), len(result[sample_name]['decompression']['real_time']))

        for i in range(0, no_of_experiment) :
            transformed_result_file.write(sample_name + ',' + result[sample_name]['compression']['real_time'][i] + ',' + result[sample_name]['compression']['cpu_time'][i] + ',' + result[sample_name]['compression']['avg_memory_used'][i] + ',' + result[sample_name]['compression']['peak_memory_used'][i] + ',' + result[sample_name]['decompression']['real_time'][i] + ',' + result[sample_name]['decompression']['cpu_time'][i] + ',' + result[sample_name]['decompression']['avg_memory_used'][i] + ',' + result[sample_name]['decompression']['peak_memory_used'][i] + '\n')

    transformed_result_file.close()

def extract_gzip_result (result_path, transformed_result_path) :
    compression_stat_file_list = glob.glob(result_path + '/gzip_compression_stat_E*')
    decompression_stat_file_list = glob.glob(result_path + '/gzip_decompression_stat_E*')
    size_stat_file_list = glob.glob(result_path + '/gzip_compression_stat_size_*')

    result = {}

    # Compressed Size Extraction
    for size_file_path in size_stat_file_list :
        size_stat_file = open(size_file_path, 'r')

        # Extract Sample Name (27 <- Trimming File Prefix, MUST BE FOLLOWED BY LENGTH OF FILE PREFIX)
        sample_name = size_file_path.split('/')[-1].split('.')[0][27:]

        result[sample_name] = {
            'compressed_size' : []
        }

        line = size_stat_file.readline()

        while line != '' :
            result[sample_name]['compressed_size'].append(int(line.replace('\n', ''))/ 10**9)
            line = size_stat_file.readline()
        size_stat_file.close()

    # Compression Time Extraction
    for compression_stat_file_path in compression_stat_file_list :
        compression_stat_file = open(compression_stat_file_path, 'r')

        # Sample Name Extraction
        sample_name = compression_stat_file_path.split('/')[-1].split('.')[0][22:]
        result[sample_name]['compression'] = {
            'real_time' : [],
            'cpu_time' : [],
            'avg_memory_used' : [],
            'peak_memory_used' : []
        }

        line = compression_stat_file.readline()

        while line != '' :
            # Skip Blank Line
            if line != '\n' :
                components = line.replace('\n', '').split(',')
                result[sample_name]['compression']['real_time'].append(components[0])
                result[sample_name]['compression']['cpu_time'].append(components[1])
                result[sample_name]['compression']['avg_memory_used'].append(components[2])
                result[sample_name]['compression']['peak_memory_used'].append(components[3])

            line = compression_stat_file.readline()
        compression_stat_file.close()
    
    # Decompression Time Extraction
    for decompression_stat_file_path in decompression_stat_file_list :
        decompression_stat_file = open(decompression_stat_file_path, 'r')

        # Sample Name Extraction
        sample_name = decompression_stat_file_path.split('/')[-1].split('.')[0][24:]
        result[sample_name]['decompression'] = {
            'real_time' : [],
            'cpu_time' : [],
            'avg_memory_used' : [],
            'peak_memory_used' : []
        }

        line = decompression_stat_file.readline()

        while line != '' :
            # Skip Blank Line
            if line != '\n' :
                components = line.replace('\n', '').split(',')
                result[sample_name]['decompression']['real_time'].append(components[0])
                result[sample_name]['decompression']['cpu_time'].append(components[1])
                result[sample_name]['decompression']['avg_memory_used'].append(components[2])
                result[sample_name]['decompression']['peak_memory_used'].append(components[3])

            line = decompression_stat_file.readline()
        decompression_stat_file.close()

    # Write Transformed Result to File
    transformed_result_file = open(transformed_result_path, 'w')
    
    # Write Header To File
    transformed_result_file.write('sample_name,compressed_size,c_real_time,c_total_cpu_time,c_avg_mem,c_peak_mem,d_real_time,d_total_cpu_time,d_avg_mem,d_peak_mem\n')

    for sample_name in result.keys() :
        for i in range(0, len(result[sample_name]['compressed_size'])):
            sample_result_record = result[sample_name]

            transformed_result_file.write(
                sample_name + ',' + 
                str(sample_result_record['compressed_size'][i]) + ',' +
                sample_result_record['compression']['real_time'][i] + ',' +
                sample_result_record['compression']['cpu_time'][i] + ',' +
                sample_result_record['compression']['avg_memory_used'][i] + ',' +
                sample_result_record['compression']['peak_memory_used'][i] + ',' +
                sample_result_record['decompression']['real_time'][i] + ',' +
                sample_result_record['decompression']['cpu_time'][i] + ',' +
                sample_result_record['decompression']['avg_memory_used'][i] + ',' +
                sample_result_record['decompression']['peak_memory_used'][i] + '\n'
            )    
    transformed_result_file.close()
def main ():
    # Arithmetic Result Extraction
    # file_size_result_file_path, time_result_file_path, transform_result_path = sys.argv[1], sys.argv[2], sys.argv[3]
    # extract_arithmetic_result(file_size_result_file_path, time_result_file_path, transform_result_path)

    # Gzip Result Extraction
    # result_file_path, transformed_result_path = sys.argv[1], sys.argv[2]
    # extract_gzip_result(result_file_path, transformed_result_path)

    # Huffman Result Extraction
    result_file_path, transformed_result_path = sys.argv[1], sys.argv[2]
    extract_huffman_result(result_file_path, transformed_result_path)

if __name__ == "__main__":
    main()