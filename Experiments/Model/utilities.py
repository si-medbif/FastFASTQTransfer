import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from Configuration import Configuration

# Data Pipeline Helper Functions
def load_seq2_seq_data (feature_file_path: str, configuration: Configuration) :
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []
    raw_score_data = []

    data_counter = 0

    with open(feature_file_path) as f:
        for line in f :
            line_items = line.rstrip().split(',')
            no_of_features = int(len(line_items)/2)

            # Baking Features
            features = line_items[:no_of_features] + [0] #Pad the last position for the attention layer
            encoder_input_data.append(features)

            raw_scores = line_items[no_of_features:]
            decoder_input_data.append([41] + raw_scores)
            decoder_target_data.append(to_categorical(raw_scores + [41], num_classes= 42))
            raw_score_data.append([int(item) for item in raw_scores])
            data_counter += 1

            if data_counter == configuration.seq_num :
                break

    encoder_input_data = np.array(encoder_input_data,dtype="float32")
    decoder_input_data = np.array(decoder_input_data,dtype="float32")
    decoder_target_data = np.array(decoder_target_data,dtype="float32")

    return encoder_input_data, decoder_input_data, decoder_target_data, raw_score_data

def generate_training_statistic_file (training_history, experiment_name, destination_file_path = 'Results/model_experiment/training_stat') : 
    # Dump History to File
    result_df = pd.DataFrame(training_history.history)
    result_df['epoch'] = [i+1 for i in range(len(result_df))]
    result_df.to_csv(destination_file_path + '/' + experiment_name + '.model_hist', index=False)

def plot_loss_acc_to_file (training_history, experiment_name, loss_chart_path='.', accuracy_chart_path='.') :
    # Loss Plot
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.plot
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(loss_chart_path + '/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

    # Accuracy Plot
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.plot
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(accuracy_chart_path + '/' + experiment_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

def encode_base(base,base_dict = {"N":0,"A":1,"T":2,"C":3,"G":4}):
  return base_dict[base.upper()]

def quality_char_to_num(ascii):
  return ord(ascii) - 33

# Model Evaluate Helper Function
def calculate_distance_from_predicted_result (actual, pred) :
    sigma_distance = 0

    for i in range(0,len(actual)) :
        sigma_distance += (actual[i] - pred[i]) ** 2

    return sigma_distance

def calculate_accuracy (pred : list, actual: list) :
  if len(pred) != len(actual) :
    print('No of Actual and Predicted did not match.')
    return -1

  no_of_item = len(pred)
  no_of_corrent = 0

  for i in range(0, no_of_item) :
    if pred[i] == actual[i] :
      no_of_corrent += 1
  
  return no_of_corrent/no_of_item

def calculate_accuracy_from_diff_file (diff_file_path: str) -> float :
  diff_file = open(diff_file_path, 'r')

  n_corrent = 0
  n_base = 0

  line = diff_file.readline()

  while line != '' :
    line_items = line[:-1].split(',')
    line_items = [int(item) for item in line_items]
    n_corrent += line_items.count(0)
    n_base += len(line_items)

    line = diff_file.readline()

  diff_file.close()

  return n_corrent/n_base
  
def offset_distribution_finder (diff_file_path: str) -> (dict, int, int) :
  diff_file = open(diff_file_path, 'r')

  offset_storage = dict()
  item_counter = 0
  read_counter = 0

  line = diff_file.readline()

  while line != '' :
    items = line[:-1].split(',')

    for item in items :
      item = int(item)
      if item not in offset_storage.keys() :
        offset_storage[item] = 1
      else :
        offset_storage[item] += 1
      item_counter += 1
    
    read_counter += 1
    line = diff_file.readline()
  
  diff_file.close()

  min_val = int(min(list(offset_storage.keys())))
  max_val = int(max(list(offset_storage.keys())))

  for val in range(min_val, max_val) :
    if val not in offset_storage.keys() :
      offset_storage[val] = 0
  
  return offset_storage, read_counter, item_counter