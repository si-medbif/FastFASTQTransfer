import sys
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from utilities import generate_training_statistic_file

# Seq2Seq Model Experiment
# INPUT: Feature File Path, Destination Hist Path, Model Path, Experiment Name
# OUTPUT: History File, Model File

class Configuration :
    def __init__ (self, latent_dim=256, num_encoder_tokens = 5, num_decoder_tokens = 41, seq_num= 10000, seq_len = 90, base_learning_rate=0.01, batch_size=10, loss='categorical_crossentropy') :
        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.base_learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.loss = loss

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
    
    training_hist = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=configuration.batch_size,
        epochs=100
    )
    
    # Save Training Stat
    generate_training_statistic_file(training_hist, experiment_name + '_1', destination_file_path = training_hist_folder_path)

    #Train the model round 2
    model.compile(
        optimizer=RMSprop(lr=configuration.base_learning_rate * 0.2), loss=configuration.loss, metrics=["accuracy"])
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=configuration.batch_size,
        epochs=500
    )

    #Train the model round 3
    model.compile(
        optimizer=RMSprop(lr=configuration.base_learning_rate * 0.2 * 0.2), loss=configuration.loss, metrics=["accuracy"])
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=configuration.batch_size,
        epochs=100
    )

    # Save Model
    model.save(model_path)        

    return model

def convert2Q (seq) :
    res = [np.argmax(seq[i,:]) for i in range(seq.shape[0])]
    return res

def convert_to_decoder_model (model, configuration, decoder_model_path) :

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

    decoder_model.save(decoder_model_path)
    
    return decoder_model

def predict_single_record (feature_file_path, decoder_model, configuration) :
    
    encoder_input_data, decoder_input_data, decoder_target_data = load_data_from_file(feature_file_path, configuration)
    target = convert2Q(decoder_target_data[777,:,:])

    decode_sequence_input_seq = encoder_input_data [777:777+1]

    # Encode the input as state vectors.
    states_value = decoder_model.predict(decode_sequence_input_seq)

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

    pred = decoded_sentence
    diff_array = np.subtract(target[:configuration.seq_len],pred[:configuration.seq_len]) #This will be used for the final correction of Q-scores

    return pred, diff_array

def main(args) :
    feature_file_path = args[1]
    configuration = Configuration()

    encoder_model_full_path = args[3] + '/' + args[4] + '_encoder.h5'
    decoder_model_full_path = args[3] + '/' + args[4] + '_decoder.h5'

    encoder_model = generate_encoder_model(feature_file_path, configuration, args[2], encoder_model_full_path, args[4])
    decoder_model = convert_to_decoder_model (encoder_model, configuration, decoder_model_full_path) 
    predict_single_record(feature_file_path, decoder_model, configuration)
    
if __name__ == "__main__":
    main(sys.argv)