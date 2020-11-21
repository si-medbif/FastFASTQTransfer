import sys
import math

# Convert dynamic size feature file to static feature file
# INPUT : feature_file_path
# OUTPUT: converted_feature_file_path

def main (args) :
    input_feature_file = open(args[1], 'r')
    converted_feature_file = open(args[2], 'w')

    for line in input_feature_file :
        line_components = line[:-1].split(',')

        # Feature half of all features set
        feature_size = math.floor(len(line_components) / 2)

        X = line_components[:feature_size]
        converted_Y = [chr(int(item) + 33) for item in line_components[feature_size:]]

        feature_set = X+converted_Y
        raw_feature = ''

        for item in feature_set :
            raw_feature += item + '\t'
        
        raw_feature = raw_feature[:-1] + '\n'

        # Write to Destination File
        converted_feature_file.write(raw_feature)

    input_feature_file.close()
    converted_feature_file.close()

if __name__ == "__main__":
    main(sys.argv)