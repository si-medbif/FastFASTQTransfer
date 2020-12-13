import sys
import pandas as pd

# Merge result file from each method into one file for visualising the result
# INPUT: <Result File> <Merged File Path>
# OUTPUT: <Merged File>

def main(args) :
    result_file_list = args[1:-1]
    destination_file = args[-1]

    new_df = pd.DataFrame()

    for result_file_path in result_file_list :
        result_df = pd.read_csv(result_file_path)
        new_df = new_df.append(result_df)

    new_df.to_csv(destination_file, index=False)

if __name__ == "__main__":
    main(sys.argv)
