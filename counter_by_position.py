import os
import sys

# Count the quality for overall and for each position

def merge_all_dict (raw_dicts) :
    result_dict = raw_dicts[0].copy()

    for current_dict in raw_dicts[1:] :
        for score in current_dict.keys() :
            if score in result_dict :
                result_dict[score] += current_dict[score]
            else :
                result_dict[score] = current_dict[score]

    return result_dict

def write_dict_to_csv(source_dict, desitnation_file_path) :
    destination_file = open(desitnation_file_path, 'w')

    for source_key in source_dict.keys() :
        destination_file.write(str(ord(source_key)-33) + ',' + str(source_dict[source_key]) + '\n')

    destination_file.close()

def main (args) :
    source_file = open(args[1], 'r')
    
    # Init Dict
    result_dict_list = []
    for i in range(0,100) :
        result_dict_list.append(dict())

    line = source_file.readline().strip()

    while line != None and line != '' :
        counter = 0

        for score_position in line :
            # Save for each position from 0-99
            if score_position in result_dict_list[counter] :
                result_dict_list[counter][score_position] += 1
            else:
                result_dict_list[counter][score_position] = 0
            counter += 1
        
        line = source_file.readline().strip()

    source_file.close()

    all_position_dict = merge_all_dict(result_dict_list)

    write_dict_to_csv(all_position_dict, args[2] + '/combined.result')

    current_position = 1

    for result_dict in result_dict_list :
        write_dict_to_csv(result_dict, args[2] + '/' + str(current_position) + '.result')
        current_position += 1

if __name__ == "__main__":
    main(sys.argv)    