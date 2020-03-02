from joblib import Parallel, delayed
from PIL import Image

import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import math

def concat_plot (source_path, destination_file) :
    image_files = list()

    for score in range(0,41) :
        if os.path.isfile(source_path + "/" + str(score) + '.png'):
            image_files.append(Image.open(source_path + "/" + str(score) + '.png'))
            print(source_path + "/" + str(score) + '.png')
        else :
            print(source_path + "/" + str(score) + ".png not found")
        
    images_width, images_height = image_files[0].width, image_files[0].height

    # Find Grid Size
    grid_size = math.ceil(math.sqrt(len(image_files)))
    final_image_file = Image.new('RGB', (images_width*grid_size, images_height*grid_size))
    
    counter = 0
    for i in range(0,grid_size) :
        for j in range(0,grid_size) :

            # The Final Merged Plot might not be perfect square format -> must break when out of images
            if counter > len(image_files)-1 :
                return

            final_image_file.paste(image_files[counter], (j*images_width, i*images_height))
            counter += 1

    final_image_file.save(destination_file)

def transform_dictionary_to_dataframe (input_dict) :
    return pd.DataFrame(data={'position' : list(input_dict.keys()), 'frequency' : list(input_dict.values())})

def plot_score_distribution (source_path, destination_path, score) :
    # Map Position to Frequency
    score_dict = dict()

    # Read Through Position Dict Files
    for position in range(1,101) :
        position_dict_file = open(source_path + '/' + str(position), 'rb')
        position_dict = pickle.load(position_dict_file)
        position_dict_file.close()
        
        if score not in position_dict :
            score_dict[position] = 0
        else :
            score_dict[position] = position_dict[score]
        
    try :
        # Store on persistence file
        persistence_file = open('score_process/position_plot_dict/' + str(score), 'wb')
        pickle.dump(score_dict, persistence_file)

        score_df = transform_dictionary_to_dataframe(score_dict)
        fig = plt.figure(figsize=(20,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_yscale('log', nonposy='clip')

        dist_plot = sns.barplot(data=score_df, x='position', y='frequency').set_title("Score : " + str(score)) 

        dist_plot.figure.savefig(destination_path + '/' + str(score) + '.png')
    except :
        return

def main (args) :
    source_path = args[1]
    destination_path = args[2]

    os.system('mkdir ' + destination_path)
        
    Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(plot_score_distribution)(source_path, destination_path, score)
        for score in range(0,41)
    )

    concat_plot(destination_path, 'score_process/final.png')


if __name__ == "__main__":
    # RUN : python3 plot_score_based.py <source_path> <destination_path>
    main(sys.argv)