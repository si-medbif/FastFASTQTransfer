import sys
import pickle

# Transform Nucleotide into Numeric Format
# INPUT: Read File, Result File Path
# OUTPUT: Numeric Format of Nucleotide

def main (args) :
    read_file = open(args[1], 'r')
    result_file = open(args[2], 'w')
    
    line = read_file.readline()
    
    while line != '' :
        nucleotides = line.rstrip()
        
        current_list = list()
        
        for base in nucleotides :
            if base == 'A':
                current_list.append(1)
            elif base == 'T' :
                current_list.append(2)
            elif base == 'C' :
                current_list.append(3)
            elif base == 'G' :
                current_list.append(4)
            elif base == 'N' :
                current_list.append(0)

        result_file.write(str(current_list)[1:-1] + '\n')
        
    result_file.close()
    read_file.close()

if __name__ == 'main' :
    main(sys.argv)