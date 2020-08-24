from joblib import Parallel, delayed

import heapq
import os
import sys
import pickle
import glob
import bitarray
import time

def get_huffman_encode_scheme(frequency):
    # This function is modified from https://gist.github.com/nboubakr/0eec4ea650eeb6dc21f9
    heap = [[weight, [symbol, '']] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def transform_list_to_dict (input_list) :
    result_dict = dict()

    for item in input_list :
        result_dict[item[0]] = item[1]

    return result_dict

def get_frequency_form_file (data_path) :
    frequency_dict = dict()
    data_file = open(data_path, 'r')

    line = data_file.readline()

    while line != None and line != "" :
        line = line.strip()
        for score in line :
            if score not in frequency_dict :
                frequency_dict[score] = 0
            frequency_dict[score] += 1

        line = data_file.readline()

    data_file.close()
    return frequency_dict

def get_frequency (data) :
    frequency_dict = dict()

    for score in data :
        if score not in frequency_dict :
            frequency_dict[score] = 0
        frequency_dict[score] += 1
    
    return frequency_dict

def merge_frequency_dict(frequency_dicts) :
    result_dict = frequency_dicts[0].copy()

    # There is only one record input
    if len(frequency_dicts) == 1 :
        return result_dict

    for current_dict in frequency_dicts[1:] :
        for key in current_dict.keys() :
            if key not in result_dict :
                result_dict[key] = 0
            result_dict[key] += current_dict[key]
    
    return result_dict

def measure_file_size_in_folder (source_folder) :
    file_list = glob.glob(source_folder + '/*')

    final_size = 0

    for file_name in file_list :
        final_size += os.path.getsize(file_name)

    return final_size

def huffman_encode_from_input (scheme, data) :
    raw_bin = ""

    for score in data :
        raw_bin += scheme[score]
    
    return bitarray.bitarray(raw_bin)

def huffman_encode_form_file (scheme, input_file_name, destination) :
    input_file = open(input_file_name, 'r')
    encoded_bit = bitarray.bitarray()
    raw_bin = ""
    line = input_file.readline()

    while line != None and line != "" :
        line = line.strip()
        raw_bin = ""
        
        for score in line :
            raw_bin += scheme[score]

        encoded_bit.extend(raw_bin)

        if len(encoded_bit) % 4096 == 0 :
            # Write to File
            destination_file = open(destination, 'ab')
            encoded_bit.tofile(destination_file)
            destination_file.close()

            # Reset the val
            encoded_bit = bitarray.bitarray()
        
        line = input_file.readline()

   
    input_file.close()
    destination_file = open(destination, 'ab')
    encoded_bit.tofile(destination_file)
    destination_file.close()

def measure_huffman_encode_quality_whole_file (input_file_name) :
    frequencies_from_chunk = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(get_frequency_form_file)(file_name)
        for file_name in glob.glob(input_file_name + "/*")
    )

    merged_frequency = merge_frequency_dict(frequencies_from_chunk)

    scheme_file_name = 'merged_frequency_dict.dump'
    dump_file = open(scheme_file_name, 'wb')
    pickle.dump(merged_frequency, dump_file)
    dump_file.close()

    encode_scheme = transform_list_to_dict(get_huffman_encode_scheme(merged_frequency))

    Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(huffman_encode_form_file)(encode_scheme, file_name, 'encoded/' + file_name.split('/')[-1].split('.')[0])
        for file_name in glob.glob(input_file_name + "/*")
    )

    destination_files = glob.glob('encoded/*')
    combined_size = 0
    for destination_file in destination_files :
        combined_size += os.path.getsize(destination_file)
    
    # Full Byte Required
    combined_size += combined_size%8

    print('Whole File Compression')
    print('Scheme File Size : ' + str(os.stat(scheme_file_name).st_size) + ' byte(s)')
    print('Data Only File Size : ' + str(combined_size/8) + ' byte(s)')

    
def measure_huffman_encode_quality_read_size (input_file_name) :
    input_file = open(input_file_name, 'r')

    line = input_file.readline()
    whole_bit_counter = 0
    whole_scheme_size = 0
    read_counter = 0

    for line in input_file :
        if line == None or len(line) == 0 :
            continue

        line = line.strip()
        
        scheme = transform_list_to_dict(get_huffman_encode_scheme(get_frequency(line)))

        for score in line :
            whole_bit_counter += len(scheme[score])

        # Counter
        whole_scheme_size += len(pickle.dumps(scheme))
        read_counter += 1

    input_file.close()

    # Return whole file size in byte(s), no of processed reads and scheme size in Byte
    return [(whole_bit_counter + (whole_bit_counter % 8)) /8, whole_scheme_size, read_counter]

def measure_file_size_position (base_source_path, position) :
    scheme = transform_list_to_dict(get_huffman_encode_scheme(get_frequency_form_file(base_source_path + '/' + str(position))))

    scheme_size = pickle.dumps(scheme)

    source_file = open(base_source_path + '/' + str(position), 'r')
    line = source_file.readline().strip()
    bit_length = 0
    while line != None and line != "" :
        bit_length += len(huffman_encode_from_input(scheme, line))
        line = source_file.readline().strip()

    source_file.close()

    return [(bit_length + (bit_length%8)) / 8, scheme_size]

def main (args) :

    # Full File Compression
    start_time = time.time()
    measure_huffman_encode_quality_whole_file(args[1])
    print('Elapsed : ' + str(time.time() - start_time) + ' second(s)')

    # Per Read Compression
    start_time = time.time()

    chunks = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(measure_huffman_encode_quality_read_size)(file_name)
        for file_name in glob.glob(args[1] + "/*")
    )

    no_of_rec = 0
    whole_read_size = 0
    whole_scheme_size = 0

    for chunk in chunks :
        whole_read_size += chunk[0]
        whole_scheme_size += chunk[1]
        no_of_rec += chunk[2]

    avg_read_size = (whole_read_size + whole_scheme_size) / no_of_rec

    print('Per Read Compression')
    print('There are ' + str(no_of_rec) + ' reads with avgerage size of ' + str(avg_read_size) + ' Byte(s) included with scheme')
    print('All Scheme Size : ' + str(whole_scheme_size) + ' Byte(s)')
    print('Data Only Size : ' + str(whole_read_size) + ' Byte(s)')
    print('Whole Data Size : ' + str(whole_read_size + whole_scheme_size) + ' Byte(s)')
    print('Elapsed : ' + str(time.time() - start_time) + ' second(s)')

    # Per Position Compression
    # start_time = time.time()
    # file_sizes = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
    #     delayed(measure_file_size_position)(args[2], position)
    #     for position in range(1,101)
    # )

    # print(file_sizes)

    # whole_data_size = 0
    # whole_scheme_size = 0
    # for file_size in file_sizes :
    #     whole_data_size += file_size[0]
    #     whole_scheme_size += file_size[1]
    
    # avg_size_data = whole_data_size / len(file_sizes)
    # avg_size_scheme = whole_scheme_size / len(file_sizes)
    # avg_size_whole = (whole_data_size + whole_scheme_size) / len(file_sizes)
    
    # print('Per Position Compression')
    # print('Average Position Based Size (included scheme) : ' + str(avg_size_whole) + ' Byte(s)')
    # print('Avgerage Score Size :' + str(avg_size_data) + ' Byte(s)')
    # print('Avgerage Scheme Size : ' + str(avg_size_scheme) + ' Byte(s)')
    # print('Data Size : ' + str(whole_data_size) + ' Byte(s)')
    # print('Scheme Size : ' + str(whole_scheme_size) + ' Byte(s)')
    # print('Full Size : ' + str(whole_data_size + whole_scheme_size) + ' Byte(s)')
    # print('Elapsed : ' + str(time.time() - start_time) + ' second(s)')

if __name__ == "__main__":
    main(sys.argv)

# Result
# Per Read : There are 1895408244 reads with avgerage size of 226.66822598931358 bits(s) Full Size : 429628824194 bit(s)
# Per Position : Average Position Based Size : 8452145953.56 bits(s) | Full Size : 845214595360 bits(s)
# [8160777932, 8345553159, 8286005266, 8259779185, 8274950185, 8292291112, 8284796765, 8289216209, 8290923131, 8304620458, 8309994345, 8338732945, 8314147956, 8318491167, 8313560059, 8325327516, 8341932122, 8333902720, 8342785301, 8357296118, 8354679273, 8345424709, 8363901944, 8366130529, 8383122987, 8371321522, 8364416229, 8346175195, 8352292536, 8356709381, 8373206259, 8369891395, 8367616979, 8361544391, 8379637471, 8376642847, 8372433028, 8381557410, 8380145749, 8379513714, 8390125484, 8382115937, 8398753304, 8408128628, 8404171751, 8413346423, 8411458833, 8423897328, 8430892481, 8429596659, 8438378930, 8420631885, 8415332786, 8429161886, 8432053413, 8450872046, 8441743735, 8450054375, 8456924497, 8454470253, 8462619312, 8451243996, 8456905494, 8470559384, 8465678319, 8460524236, 8474389188, 8486308315, 8482873331, 8498145489, 8508909179, 8540899683, 8521689190, 8523130392, 8540741018, 8542629312, 8525484288, 8600424498, 8567514098, 8567530184, 8558968098, 8621756517, 8599354658, 8599672863, 8601774377, 8600675138, 8632140917, 8636127442, 8643592948, 8658238060, 8647921771, 8663525736, 8683343651, 8656287709, 8695165336, 8693248729, 8692599445, 8731460229, 8709735438, 8827249555]