import sys
import pickle

# Generate Quality Score Commulative Score
# INPUT: Quality Score Frequency Dictionary Pickle File, Result File Path
# OUTPUT: Commulative Score Pickle File

def main (args) :
    # Load Score Frequency Pickle File
    score_freq_file = open(args[1], 'rb')
    score_freq = pickle.load(score_freq_file)
    score_freq_file.close()
    
    frequency_list = list(score_freq.items())
    sorted_frequency_list = sorted(frequency_list, reverse=True)
    
    no_of_items = sum(sorted_frequency_list)
    
    commulative_list = {'Order' : [], 'Percentage' : []}
    
    for count, value in enumerate(sorted_frequency_list) :
        commulative_list['Order'].append(count)
        commulative_list['Percentage'].append((value/no_of_items)*100)

    # Dump result into file
    result_file = open(args[2], 'wb')
    pickle.dump(commulative_list, result_file)
    result_file.close()
    
if __name__ == 'main' :
    main(sys.argv)