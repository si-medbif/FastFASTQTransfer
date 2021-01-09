import pandas as pd
import matplotlib.pyplot as plt

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