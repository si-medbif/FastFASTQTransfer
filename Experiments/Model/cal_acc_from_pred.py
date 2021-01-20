from joblib import Parallel, delayed

import sys
import glob

from utilities import calculate_accuracy_from_diff_file

# Bidirectional LSTM with Dot Attention Seq2Seq Model Experiment
# INPUT: Subtracted Pred File Path 
# OUTPUT: Accuracy for each experiment

def result_maker (file_path) :
    accuracy = calculate_accuracy_from_diff_file(file_path)
    print(file_path.split('/')[-1].split('.')[0], accuracy)

def main (args) :
    file_list = glob.glob(args[1] + '/*')

    Parallel(n_jobs=-1, prefer="processes", verbose=0)(
        delayed(result_maker)(file_path)
        for file_path in file_list
    )
    
if __name__ == "__main__":
    main(sys.argv)