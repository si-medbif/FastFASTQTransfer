from joblib import Parallel, delayed

import glob
import os

# Data Preprocessing (Data Extraction)

def get_read_extraction_from_fastq_jobs (fastq_path, read_path) :
    fastq_file_path_list = glob.glob(fastq_path + '/*.fastq')
    return ["awk '0 == (NR + 2) % 4' " + fastq_file_path + " > " + read_path + '/' + fastq_file_path.split('.')[0] + '.read' for fastq_file_path in fastq_file_path_list]

def get_quality_score_extraction_from_fastq_jobs (fastq_path, quality_path) :
    fastq_file_path_list = glob.glob(fastq_path + '/*.fastq')
    return ["awk '0 == (NR + 4) % 4' " + fastq_file_path + " > " + quality_path + '/' + fastq_file_path.split('.')[0] + '.quality' for fastq_file_path in fastq_file_path_list]

def extract_read_and_quality_from_fastq (fastq_path, read_path, quality_path) :
    jobs = get_read_extraction_from_fastq_jobs(fastq_path, read_path) + get_quality_score_extraction_from_fastq_jobs(fastq_path,quality_path)
    
    # Parallelly Extract Read and Quality Score from FASTQ File By using AWK
    Parallel(n_jobs=-1, prefer="processes", verbose=0)(
            delayed(os.system)(command_job)
            for command_job in jobs
    )