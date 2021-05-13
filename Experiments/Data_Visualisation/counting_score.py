import sys
import pickle 

# Generate Quality Score Frequency Dictionary
# INPUT: Extracted Score File Path, Result File Path
# OUTPUT: Quality Score Frequency Dictionary Pickle File

def main (args) :
    quality_file = open(args[1], 'r')
    
    line = quality_file.readline()
    FC = dict()
    
    while line != '' :
        
        line = quality_file.readline().rstrip()
        for score_char in line :
            if score_char not in FC :
                FC[score_char] = 1
            else :
                FC[score_char] += 1
    
    quality_file.close()
    
    # Dump FC into file
    output_file = open(args[2], 'wb')
    pickle.dump(FC, output_file)
    output_file.close()

if __name__ == 'main' :
    main(sys.argv)