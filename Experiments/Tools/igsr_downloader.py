import sys
import os
import wget
import hashlib
import pandas as pd

from joblib import Parallel, delayed

def prepare_path (destination_folder) :
    if not os.path.exists(destination_folder) :
        os.makedirs(destination_folder)

def transform_df_to_list (df) :
    result = list()
    
    for index, row in df.iterrows():
        result.append({'url' : row['url'], 'md5' : row['md5']})

    return result

def check_md5 (file_name, original_md5) :
    if original_md5 == hashlib.md5(open(file_name, 'rb').read()).hexdigest() :
        print(file_name.split('/')[0], ' Integrity Check Passed')
        return 0
    else :
        print(file_name.split('/')[0], ' Integrity Check Failed')
        return -1

def unpack_compressed_file (source_file, destination_file) :
    print('Decompressing', source_file.split('/')[-1])
    os.system('gzip -d ' + source_file + ' ' + destination_file)

def check_paired_end_file (destination_path, sample_list) :
    missing_file = list()

    for sample_name in sample_list :
        if not os.path.exists(destination_path + '/' + sample_name + '_1.fastq'):
            missing_file.append(destination_path + '/' + sample_name + '_1.fastq')
        
        if not os.path.exists(destination_path + '/' + sample_name + '_2.fastq'):
            missing_file.append(destination_path + '/' + sample_name + '_2.fastq')
    
    return missing_file

def write_content_to_file (source_path, destination_path) :
    source_file = open(source_path, 'r')
    destination_file = open(destination_path, 'r')
    current_line = source_file.readline()

    while current_line != '' :
        destination_file.write(current_line)
        current_line = source_file.readline()
    
    source_file.close()
    destination_file.close()

def merge_paired_end_file (source_path, destination_path, sample_id) :
    source_file_pair_1 = source_path + '/' + sample_id + '_1.fastq'
    source_file_pair_2 = source_path + '/' + sample_id + '_2.fastq'
    full_destination_path = destination_path + '/' + sample_id + '.fastq'

    write_content_to_file(source_file_pair_1, full_destination_path)
    write_content_to_file(source_file_pair_2, full_destination_path)

def download_file_from_url (url, file_destination) :
    print('Downloading ', file_destination.split('/')[-1])

    if not os.path.isfile(file_destination) :
        try:
            wget.download(url, file_destination)
        except:
            return -1
    else :
        print('Already Downloaded Skipping..')

    return 0

def main_pipeline (file_info, file_destination) :
    url = file_info['url']
    original_md5 = file_info['md5']
    file_name = url.split('/')[-1]
    full_source_path = file_destination + '/' + file_name
    
    download_file_from_url(url, full_source_path)
    check_md5(full_source_path, original_md5)
    unpack_compressed_file(full_source_path, file_destination)

    return file_name.split('_')[0]


def main (args) :
    # Handle CSV List
    input_source = args[1]
    input_file_list = pd.read_csv(input_source, sep="\t")

    # Handle Destination Path Create if not existed
    file_destination = args[2]
    prepare_path(file_destination)

    sample_list = Parallel(n_jobs=-1, prefer="processes", verbose=0)(
        delayed(main_pipeline)(file_info, file_destination)
        for file_info in transform_df_to_list(input_file_list)
    )

    missing_file = check_paired_end_file(file_destination, sample_list)
    if len(missing_file) == 0 :
        print(len(sample_list), ' sample(s) has been downloaded successfully')
    else: 
        print('The following file(s) is missing please check again', missing_file)

    # Merge Paired End File into Single File
    for sample_id in sample_list :
        merge_paired_end_file(file_destination, file_destination, sample_id)

if __name__ == "__main__":
    # RUN: python3 igsr_downloader.py [tsv_input_path] [destination_folder]
    main(sys.argv)
