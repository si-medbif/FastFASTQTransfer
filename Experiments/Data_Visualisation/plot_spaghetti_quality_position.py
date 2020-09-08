import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def convert_line_to_score_list(line) :
    result = []
    for score in line :
        result.append(ord(score) - 33)

    return result

def main (args) :
    input_file = open(args[1] , 'r')
    x = pd.Series(list(range(100)))

    for line in input_file :
        y = pd.Series(convert_line_to_score_list(line))
        frame = {'Position' : x, 'Score' : y}
        sns.lineplot(data=pd.DataFrame(frame), x="Position", y="Score", alpha=0.03, color='blue')

    input_file.close()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)