import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from utilities import generate_training_statistic_file, calculate_distance_from_predicted_result, offset_distribution_finder
from Configuration import Configuration

# Seq2Seq Model Experiment
# INPUT: Feature File Path
# OUTPUT: History File, Model File, Array Diff Result

def load_data_from_file (feature_file_path, configuration) :
    encoder_input_data = np.zeros((configuration.seq_num, configuration.seq_len, configuration.num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((configuration.seq_num, configuration.seq_len + 1, configuration.num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((configuration.seq_num, configuration.seq_len + 1, configuration.num_decoder_tokens), dtype='float32')

    with open(feature_file_path, 'r') as f:
        record_index = 0
        for line in f :
            features = list(map(int,line.rstrip().split(",")))

            # Process Input (HotEncoder)
            for position in range(configuration.seq_len) :
                encoder_input_data[record_index, position, features[position]] = 1.0

            # Process Y
            for position in range(configuration.seq_len, len(features)) :
                decoder_input_data[record_index, position+1-configuration.seq_len, features[position]] = 1.0
                decoder_target_data[record_index, position-configuration.seq_len, features[position]] = 1.0

            decoder_input_data[record_index, 0, 40] = 1.0
            decoder_target_data[record_index, configuration.seq_len, 40] = 1.0

            record_index += 1

            # Load only use
            if record_index == configuration.seq_num :
                break
            
    return encoder_input_data, decoder_input_data, decoder_target_data

def generate_encoder_model (feature_file_path, configuration, training_hist_folder_path, model_path, experiment_name) :
    encoder_input_data, decoder_input_data, decoder_target_data = load_data_from_file(feature_file_path, configuration)
        
    # Build a model

    # Define an input sequence and process it
    encoder_inputs = Input(shape=(None, configuration.num_encoder_tokens))
    encoder = LSTM(configuration.latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, configuration.num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(configuration.latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(configuration.num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #Train the model round 1
    model.compile(
        optimizer=RMSprop(lr=configuration.base_learning_rate), loss=configuration.loss, metrics=["accuracy"])
    
    training_hist_1 = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=configuration.batch_size,
        epochs=100
    )

    #Train the model round 2
    model.compile(
        optimizer=RMSprop(lr=configuration.base_learning_rate * 0.2), loss=configuration.loss, metrics=["accuracy"])
    training_hist_2 = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=configuration.batch_size,
        epochs=500
    )

    #Train the model round 3
    model.compile(
        optimizer=RMSprop(lr=configuration.base_learning_rate * 0.2 * 0.2), loss=configuration.loss, metrics=["accuracy"])
    training_hist_3 = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=configuration.batch_size,
        epochs=100
    )

    # Merge Training Hist
    training_hist_1 = pd.DataFrame(training_hist_1.history) #100 Epoch
    training_hist_2 = pd.DataFrame(training_hist_2.history) #500 Epoch
    training_hist_3 = pd.DataFrame(training_hist_3.history) #100 Epoch

    training_hist_1['epoch'] = [i+1 for i in range(len(training_hist_1))]
    training_hist_2['epoch'] += [i+1+100 for i in range(len(training_hist_2))] # 100
    training_hist_3['epoch'] += [i+1+600 for i in range(len(training_hist_3))] # 100 # 100 + 500

    merged_result = pd.concat([training_hist_1, training_hist_2, training_hist_3])
    
    # Save Training Stat
    merged_result.to_csv(training_hist_folder_path + '/' + experiment_name + '.model_hist', index=False)

    # Save Model
    model.save(model_path)        

    return model

def convert2Q (seq) :
    res = [np.argmax(seq[i,:]) for i in range(seq.shape[0])]
    return res

def convert_to_decoder_model (model, configuration, decoder_model_path, write_model_to_file=True) :

    # Load model from file if model path is specified : otherwise model instance is accquired
    if type(model) == str :
        model = load_model(model)

    # Transform Model
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(configuration.latent_dim,), name="input_3")
    decoder_state_input_c = Input(shape=(configuration.latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm (
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model (
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    if write_model_to_file :
        decoder_model.save(decoder_model_path)
    
    return encoder_model, decoder_model

def predict_from_file (feature_file_path, encoder_model_path, decoder_model_path, configuration, array_diff_full_path, mse_log_full_path):
    encoder_input_data, decoder_input_data, decoder_target_data = load_data_from_file(feature_file_path, configuration)
    
    no_of_process = 16
    chunk_size = int(configuration.seq_num / no_of_process)

    diff_result_containers = Parallel(n_jobs=no_of_process, prefer="processes")(
            delayed(predict_from_data)(encoder_input_data[chunk_size*chunk_index: chunk_size*(chunk_index+1)], decoder_input_data[chunk_size*chunk_index: chunk_size*(chunk_index+1)], encoder_model_path, decoder_model_path, configuration)
            for chunk_index in range(int(configuration.seq_num / no_of_process))
    )

    # Normalise to single container
    # Before: [ [[0,0,-1],[0,-4,2]], [[0,0,-1],[0,-4,2]] ]
    # After: [ [0,0,-1], [0,-4,2], [0,0,-1], [0,-4,2] ]
    diff_result_container = list()

    for result_container in diff_result_containers :
        for diff_read in result_container :
            diff_result_container.append(diff_read)
    del diff_result_containers

    # Creating Result File (MSE Log and Diff File)
    mse_progress_file = open(mse_log_full_path, 'w')
    diff_result_file = open(array_diff_full_path, 'w')

    # Calculate MSE and Accuracy
    correct_predicted_counter = 0
    result_sigma = 0
    total_base = 0

    for diff_read in diff_result_container :
        correct_predicted_counter += diff_read.count(0)
        result_sigma += np.sum(np.power(np.array(diff_read), 2))
        total_base += len(diff_read)

        mse = (1/total_base) * result_sigma
        accuracy = correct_predicted_counter / total_base

        mse_progress_file.write(str(mse) + '\n')
        diff_result_file.write(str(diff_read)[1:-1].replace(' ', '') + '\n')

    mse_progress_file.close()
    diff_result_file.close()

    return mse, accuracy

def predict_from_data (encoder_input_data, decoder_target_data, encoder_model_path, decoder_model_path, configuration) :

    # Load Model
    encoder_model, decoder_model = convert_to_decoder_model (encoder_model_path, configuration, decoder_model_path, write_model_to_file=False) 

    diff_result_container = list()

    for data_index in range(0, len(decoder_target_data)) :
        target = convert2Q(decoder_target_data[data_index,:,:])

        decode_sequence_input_seq = encoder_input_data [data_index:data_index+1]

        # Encode the input as state vectors.
        states_value = encoder_model.predict(decode_sequence_input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, configuration.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, 40] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence =[]#= ""

        while not stop_condition :
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            #sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence.append(sampled_token_index) #+= sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sentence) > configuration.seq_len + 1:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, configuration.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]

        # Calcualte diff
        diff_array = np.subtract(target[:configuration.seq_len],decoded_sentence[:configuration.seq_len]) #This will be used for the final correction of Q-scores

        # Append to result container
        diff_result_container.append(diff_array.tolist())
    
    return diff_result_container

def plot_diff_distribution (diff_file_path, experiment_name:str, configuration: Configuration) :
    diff_dist, no_of_read, no_of_item = offset_distribution_finder(diff_file_path)

    # Normalise into percentage
    diff_list = list()
    percentages_list = list()
    
    for diff in diff_dist.keys() :
        if diff == 0:
            continue
        diff_list.append(diff)
        percentages_list.append(diff_dist[diff]/ no_of_item * 100)
    
    diff_dist_percentage = pd.DataFrame({'Offset': diff_list, 'Percentage' : percentages_list}).sort_values(by=['Offset'])
    
    sns.barplot(data=diff_dist_percentage, x='Offset', y='Percentage')
    plt.title('Prediction offset ' + experiment_name + ' (' + str(configuration.seq_num) + ' reads)')
    plt.show()

def main(args) :
    # Feature File Path, Destination Hist Path, Model Path, Array Diff Path, Experiment Name

    latent_dim = 1024
    batch_size = 100
    base_learning_rate = 0.001
    seq_num=10000

    feature_file_path = args[1]
    destination_training_hist_path = 'Results/model_experiment/training_stat/seq2seq'
    model_path = 'Results/model_experiment/model/seq2seq'
    array_diff_path = 'Results/model_experiment/predicted_diff'
    mse_progress_log_path = 'Results/model_experiment/mse_log'
    experiment_name = 'seq2seq_L' + str(latent_dim) + '_Lr'+ str(base_learning_rate).replace('.', '-') + '_BS' + str(batch_size) + '_' + str(seq_num) 

    configuration = Configuration(
        experiment_name = 'Seq2Seq',
        latent_dim=latent_dim,
        batch_size=batch_size,
        base_learning_rate=base_learning_rate,
        seq_num=seq_num
    )

    encoder_model_full_path = model_path + '/' + experiment_name + '_encoder.h5'
    decoder_model_full_path = model_path + '/' + experiment_name + '_decoder.h5'
    array_diff_full_file_name = array_diff_path + '/' + experiment_name + '.diff'
    mse_log_full_file_name = mse_progress_log_path + '/' + experiment_name + '_MSE.csv'

    # encoder_model = generate_encoder_model(feature_file_path, configuration, destination_training_hist_path, encoder_model_full_path, experiment_name)

    # encoder_model_path = 'Results/model_experiment/model/seq2seq/seq2seq_L1024_Lr0-001_BS100_10000_encoder.h5'
    # encoder_model, decoder_model = convert_to_decoder_model (encoder_model_path, configuration, decoder_model_full_path) 
    # mse, accuracy = predict_from_file(feature_file_path, encoder_model_path, decoder_model_full_path, configuration, array_diff_full_file_name, mse_log_full_file_name)
    # print('MSE:', mse, 'Accuracy:', accuracy)
    plot_diff_distribution(array_diff_full_file_name, experiment_name, configuration)
    
if __name__ == "__main__":
    main(sys.argv)