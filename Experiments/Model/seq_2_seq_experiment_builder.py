import sys
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from utilities import generate_training_statistic_file

# Seq2Seq Model Experiment
# INPUT: Feature File Path, Destination Hist Path, Model Path, Experiment Name
# OUTPUT: History File, Model File

class Configuration :
    def __init__ (self, latent_dim=256, num_encoder_tokens = 5, num_decoder_tokens = 41, seq_num= 100000, seq_len = 90) :
        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.seq_num = seq_num
        self.seq_len = seq_len

def generate_model (feature_file_path, configuration, training_hist_folder_path, model_path, experiment_name) :
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
            optimizer=RMSprop(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
        
        training_hist = model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=10,
            epochs=100
        )
        
        # Save Training Stat
        generate_training_statistic_file(training_hist, experiment_name + '_1', destination_file_path = training_hist_folder_path)

        #Train the model round 2
        model.compile(
            optimizer=RMSprop(lr=0.01 *0.2), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=10,
            epochs=500
        )

        #Train the model round 3
        model.compile(
            optimizer=RMSprop(lr=0.01 *0.2 * 0.2), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=10,
            epochs=100
        )

        # Save Model
        model.save(model_path + '/' + experiment_name + '.h5')        

def main(args) :
    feature_file_path = args[1]
    configuration = Configuration(seq_num=300000)

    generate_model(feature_file_path, configuration, args[2], args[3], args[4])

if __name__ == "__main__":
    main(sys.argv)