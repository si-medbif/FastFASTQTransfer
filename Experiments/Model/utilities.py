import matplotlib.pyplot as plt

def generate_training_statistic_file (training_history, experiment_name, destination_file_path = 'Results/model_experiment/training_stat') : 

    # Dump History to File
    model_hist_file = open(destination_file_path + '/' + experiment_name + '.model_hist', 'w')
    model_hist_file.write('loss,val_loss,acc,val_acc\n')
    
    for i in range(0,len(training_history.history['loss'])) :
        model_hist_file.write(str(training_history.history['loss'][i]) + ',' + str(training_history.history['val_loss'][i]) + ',' + str(training_history.history['accuracy'][i]) + ',' + str(training_history.history['val_accuracy'][i]) + '\n')

    model_hist_file.close()

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