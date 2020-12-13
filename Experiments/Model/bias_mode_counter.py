import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plot count quality score that equal to MODE of each position
# INPUT: <Feature File Path> <No of Row to process>
# OUTPUT: <Percentage of reads that equal to mode>

def count_percentage (col_list, mode) :
    counter = 0
    no_of_data = len(col_list)

    for item in col_list :
        if item == mode :
            counter += 1
    
    return (counter/no_of_data) * 100

def main(args) :
    dataset = pd.read_csv(args[1], header=None, nrows=int(args[2])).iloc[:,90:]
    percentage_list = list()

    for col_name in dataset.columns :
        percentage_list.append(count_percentage(list(dataset[col_name]), dataset[col_name].mode().tolist()[0]))

    a = []
    for i in range(1,91) :
        a.append(i)

    sns.lineplot(x=range(1,91), y=percentage_list)
    plt.xlabel('Position')
    plt.ylabel('Percentage')
    plt.title('Percentage of quality score matches mode in each position')
    plt.show()

    sns.lineplot(y=dataset.mode().iloc[0,:].tolist(), x=range(1,91))
    plt.xlabel('Position')
    plt.ylabel('Mode')
    plt.title('Mode of quality score in each position')
    plt.show()

if __name__ == "__main__":
    main(sys.argv)