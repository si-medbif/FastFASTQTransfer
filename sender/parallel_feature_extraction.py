from joblib import Parallel, delayed
import statistics
import sys

# Extracting Feature into Feature File

def check_file_integrity (file_path, chk_record=20) :
    input_file = open(file_path, 'r')
    record_length = []

    current_record_length = 0
    previous_line = ""
    while True:
        line = input_file.readline()
        current_record_length += len(line)

        # Trim New Line Character
        if line[-1] == "\n" :
            line = line[:-1]

        previous_line = line

        if previous_line == "+" :
            record_length.append(current_record_length)
            current_record_length = 0

        # No of record is sufficient
        if len(record_length) == chk_record :
            break  
    
    # If all of the record has the same length > mean length must be equals to all record
    if statistics.mean(record_length) == record_length[0] :
        print('Non Constant Length Record Exitting...')
        exit()

    input_file.close()
    return record_length[0]

def is_eof (file_path, line_number) :
    input_file = open(file_path, 'r')

    # Get Line Size
    input_file.readline()
    line_size = input_file.tell()

    # Go To Targetted Line
    input_file.seek(line_size * line_number)
    line = input_file.readline()

    input_file.close()

    return line == ''
    
def base_mapper (base) :
    if base.upper() == 'A' :
        return 1
    elif base.upper() == 'T' :
        return 2
    elif base.upper() == 'C' :
        return 3
    elif base.upper() == 'G' :
        return 4
    else :
        # Unknown Base (N)
        return 0

def read_n_line (file_path, start_line, end_line) :
    if start_line > end_line :
        return None

    # Open File (File MUST contain equal length per line)
    input_file = open(file_path, 'r')

    # Find Line Size
    input_file.readline()
    line_count = input_file.tell()
    
    # Go To Start Line
    input_file.seek((start_line-1) * line_count)
    read_line = input_file.read((line_count-1) * (end_line-start_line))

    # EOF Reached
    if read_line == "" :
        return None

    input_file.close()

    return read_line

def produce_feature_from_qscore (qscore_line) :
    final_feature = []

    for score in qscore_line :
        # Convert Quality Score based on Illua Base 33 System (Illumina, Ion Torrent, PacBio and Sanger)
        final_feature.append(ord(score)- 33)
    
    return final_feature

def produce_feature_from_read (read_line) :
    final_feature = []

    for base in read_line :
        final_feature.append(base_mapper(base))
    
    return final_feature

def create_jobs (file_path, chunk_size= 1000000):
    jobs = []
    current_line = 0

    while is_eof(file_path, current_line) != True :
    # while current_line < 1896000000 :
        jobs.append((current_line +1, current_line+chunk_size))
        print('Job Created : ', current_line +1, current_line+chunk_size)
        current_line += chunk_size

    print(len(jobs), ' jobs has been created.')
    return jobs

def create_chunk_feature_file (read_file_path, qscore_file_path, feature_file_path, start_line, end_line) :
    # Get Data
    
    read_lines = read_n_line(read_file_path, start_line, end_line)
    qscore_lines = read_n_line(qscore_file_path, start_line, end_line)

    if read_lines is None or qscore_lines is None :
        print('Rejected Job')
        return
    
    read_lines = read_lines.split('\n')
    qscore_lines = qscore_lines.split('\n')

    n_line = min(len(read_lines), len(qscore_lines))
    text_to_write = ""

    for line_number in range(n_line) :
        read_line = read_lines[line_number]
        qscore_line = qscore_lines[line_number]
        if read_line == '' or qscore_line == '' :
            continue
        text_to_write += str(produce_feature_from_read(read_line) + produce_feature_from_qscore(qscore_line)).replace(" ", "")[1:-1] + '\n'
    
    feature_file = open(feature_file_path, 'a')
    feature_file.write(text_to_write)
    feature_file.flush()
    feature_file.close()

def parallel_extract_feature (read_file_path, qscore_file_path, feature_file_path, chunk_size=500000) :

    feature_file = open(feature_file_path, 'w')
    feature_file.flush()
    feature_file.close()

    Parallel(n_jobs=-1, prefer="processes", verbose=11)(
            delayed(create_chunk_feature_file)(read_file_path, qscore_file_path, feature_file_path, start_line, end_line)
            for start_line, end_line in create_jobs (read_file_path, chunk_size)
    )