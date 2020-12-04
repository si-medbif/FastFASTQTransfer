from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ordinal_categorical_crossentropy as OCC

# Plot How Model Predicted Result Shifted from Actual Score
# INPUT: Model Path, Data Path, Histogram Path (Folder)
# OUTPUT: Offset Histogram

def convert_prob_to_score (predicted_Y) :
    predicted_score = []

    for predicted_item in predicted_Y :
        predicted_score.append(np.where(predicted_item == predicted_item.max())[0][0])
    
    return predicted_score

def compare_predicted_actual_offset (predicted, actual) :

    offset = []

    for i in range(0,len(predicted)) :
        offset.append(actual[i]-predicted[i])
    
    return offset

def count_offset_occurance (offset_list) :
    counter = {}
    
    np_offset_list = np.array(offset_list)

    count_range = max(abs(np_offset_list.min()), abs(np_offset_list.max()))

    for i in range(count_range * -1, count_range+1) :
        counter[i] = 0
    
    for offset_item in offset_list : 
        counter[offset_item] += 1

    return counter

def main (args) :
    model_list = glob.glob(args[1] + '/*.h5')

    for model_file in model_list : 
        model = load_model(model_file, custom_objects={'loss' : OCC.loss})
        no_of_data = int(model_file.split('/')[-1].split('.')[0].split('_')[-1])
        dataset = pd.read_csv(args[2], nrows=no_of_data, header=None)
        
        X = dataset.iloc[:,:90]
        Y = to_categorical(dataset.iloc[:,90], 43)

        offset_list = compare_predicted_actual_offset(convert_prob_to_score(model.predict(X)), dataset.iloc[:,90])

        # Plot Histogram
        bin_size = max(abs(np.array(offset_list).min()), abs(np.array(offset_list).max()))
        sns.distplot(offset_list, kde=False, bins=range(bin_size * -1,bin_size),norm_hist=True)
        plt.title('Predicted vs Actual Offset (' + str(no_of_data) + ' reads)')
        plt.xlabel('Offset')
        plt.ylabel('Percentage')
        plt.savefig(args[3] + '/' + model_file.split('/')[-1].split('.')[0] + '.png', dpi=300)
        plt.clf()

        # Plot only Error
        non_zero_offset_list = count_offset_occurance(offset_list)
        non_zero_offset_list[0] = 0
        for score in non_zero_offset_list.keys() :
            non_zero_offset_list[score] = (non_zero_offset_list[score]/ no_of_data) * 100
        
        x = list(non_zero_offset_list.keys())
        y = list(non_zero_offset_list.values())
        
        g = sns.barplot(x=x, y=y)
        g.figure.set_size_inches(30,10)
        plt.title('Predicted vs Actual Offset (' + str(no_of_data) + ' reads) only error')
        plt.xlabel('Offset')
        plt.ylabel('Percentage')
        plt.savefig(args[3] + '/' + model_file.split('/')[-1].split('.')[0] + '_errors.png', dpi=300)
        plt.clf()

if __name__ == "__main__":
    main(sys.argv)
