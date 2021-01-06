import sys
import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

from utilities import generate_training_statistic_file, encode_base, quality_char_to_num
from Configuration import Configuration
from CustomCallbacks import WarmUpReduceLROnPlateau

# Bidirectional LSTM Seq2Seq Model Experiment
# INPUT: Feature File Path, Destination Hist Path, Model Path, Array Diff Path, MSE Progress File, Experiment Name
# OUTPUT: History File, Model File, Array Diff Result

def load_data (feature_file_path: str, configuration: Configuration) :
    encoder_input_data = []
    decoder_input_data = []
    decoder_target_data = []

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

            data_counter += 1

            if data_counter == configuration.seq_num :
                break

    encoder_input_data = np.array(encoder_input_data,dtype="float32")
    decoder_input_data = np.array(decoder_input_data,dtype="float32")
    decoder_target_data = np.array(decoder_target_data,dtype="float32")

    return encoder_input_data, decoder_input_data, decoder_target_data

def build_bidirectional_seq2seq_model (configuration: Configuration, model_full_path: str, encoder_input_data, decoder_input_data, decoder_target_data) :
    
    # Encoder
    encoder_inputs = Input(shape=(configuration.seq_len + 1,))
    encoder_embed = Embedding(output_dim=configuration.num_encoder_embed, input_dim=configuration.num_encoder_tokens)
    encoder = Bidirectional(LSTM(configuration.latent_dim, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embed(encoder_inputs))
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(configuration.seq_len + 1, ))
    decoder_embed = Embedding(output_dim=configuration.num_decoder_embed, input_dim=configuration.num_decoder_tokens)  
    decoder_lstm = LSTM(configuration.latent_dim*2, return_sequences=True, return_state=True)
    decoder_outputs,_,_= decoder_lstm(decoder_embed(decoder_inputs), initial_state=encoder_states)

    # Attention Layer
    attn_out = Attention()([encoder_outputs, decoder_outputs])

    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
    
    decoder_dense = Dense(configuration.num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # model.summary()
    reduce_lr = WarmUpReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0, init_lr = configuration.base_learning_rate, warmup_batches = configuration.count_train * 5, min_delta = 0.001)
    earlystop_callback = EarlyStopping(monitor='loss', min_delta=0.00001, patience=16)
    model.compile(optimizer=RMSprop(lr=configuration.base_learning_rate), loss="categorical_crossentropy", metrics=["accuracy","mse"])
    model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=configuration.batch_size,
    epochs=1000,
    callbacks=[reduce_lr,earlystop_callback]
    )

def main(args) :
    feature_file = args[1]
    training_hist_path = args[2]
    model_path = args[3]
    mse_progress = args[4]
    experiment_name = args[5]

    model_full_path = model_path + '/' + experiment_name + '.h5'


    configuration = Configuration(
        experiment_name = args[5],
        latent_dim=32,
        num_encoder_tokens = 5,
        num_decoder_tokens = 42,
        num_encoder_embed = 2,
        num_decoder_embed = 32,
        seq_num= 10000,
        seq_len = 90,
        base_learning_rate=0.001,
        batch_size=int(10000 * 0.01),
        loss='categorical_crossentropy'
    )

    encoder_input_data, decoder_input_data, decoder_target_data = load_data(feature_file, configuration)
    build_bidirectional_seq2seq_model(configuration, model_full_path, encoder_input_data, decoder_input_data, decoder_target_data)

if __name__ == "__main__":
    main(sys.argv)