import sys
import pickle
import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention, TimeDistributed, Dot, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from typing import Any
from utilities import load_seq2_seq_data, generate_training_statistic_file
from Configuration import Configuration
from CustomCallbacks import WarmUpReduceLROnPlateau

# Bidirectional LSTM with Dot Attention Seq2Seq Model Experiment
# INPUT: Feature File Path, Destination Hist Path, Model Path, Array Diff Path, MSE Progress File, Experiment Name
# OUTPUT: History File, Model File, Array Diff Result

def build_bidirectional_dot_attention_seq2seq_model (configuration: Configuration, model_full_path: str, training_hist_path: str, encoder_input_data, decoder_input_data, decoder_target_data) -> Model :
    
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embed = Embedding(output_dim=configuration.num_encoder_embed, input_dim=configuration.num_encoder_tokens)
    encoder = Bidirectional(LSTM(configuration.latent_dim, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embed(encoder_inputs))
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, ))
    decoder_embed = Embedding(output_dim=configuration.num_decoder_embed, input_dim=configuration.num_decoder_tokens)   
    decoder_lstm = LSTM(configuration.latent_dim*2, return_sequences=True, return_state=True)
    decoder_outputs,_,_= decoder_lstm(decoder_embed(decoder_inputs), initial_state=encoder_states)

    # Dot attention
    attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention = Activation('softmax')(attention)
    
    context = Dot(axes=[2, 1])([attention,encoder_outputs]) # dot([attention, encoder], axes=[2,1])
    decoder_combined_context = Concatenate(axis = -1)([context, decoder_outputs])

    pre_decoder_outputs = TimeDistributed(Dense(configuration.latent_dim*2, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
    decoder_dense =  TimeDistributed(Dense(configuration.num_decoder_tokens, activation='softmax'))
    decoder_outputs = decoder_dense(pre_decoder_outputs )

    # Plug everything together
    full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile and Build the model
    earlystop_callback = EarlyStopping(monitor='loss', min_delta=0.00001, patience=16)
    model_checkpoint_callback = ModelCheckpoint(filepath=model_full_path, monitor='accuracy', mode='max', save_best_only=True)
    reduce_lr = WarmUpReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0, init_lr = configuration.base_learning_rate, warmup_batches = configuration.count_train * 5, min_delta = 0.001)

    full_model.compile(optimizer=RMSprop(lr=configuration.base_learning_rate), loss=configuration.loss, metrics=["accuracy"])

    training_hist = full_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=configuration.batch_size, epochs=1000, callbacks=[reduce_lr,earlystop_callback,model_checkpoint_callback])
    
    # Save training stat to file
    generate_training_statistic_file(training_hist, configuration.experiment_name, destination_file_path = training_hist_path)

    return full_model

def transform_model (full_model, configuration: Configuration) -> (Model, Model, Model) :
    if type(full_model) == str :
        full_model = load_model(full_model)
    
    # Encoder Model
    inf_enc_input = full_model.input[0]
    inf_enc_embed = full_model.layers[1]
    inf_encoder = full_model.layers[3]
    inf_enc_out, inf_fh, inf_fc, inf_bh, inf_bc =  inf_encoder(inf_enc_embed(inf_enc_input))
    inf_enc_state_h = Concatenate()([inf_fh, inf_bh])
    inf_enc_state_c = Concatenate()([inf_fc, inf_bc])
    inf_enc_states = [inf_enc_state_h, inf_enc_state_c]

    encoder_model = Model(inf_enc_input,[inf_enc_out]+inf_enc_states)

    # Decoder Model
    decoder_inputs = full_model.input[1]  # input_2
    decoder_embed_inputs = full_model.layers[4](decoder_inputs)

    decoder_state_input_h = Input(shape=(configuration.latent_dim * 2,), name="input_3")
    decoder_state_input_c = Input(shape=(configuration.latent_dim * 2,), name="input_4")

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = full_model.layers[7]
    pre_decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embed_inputs , initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [pre_decoder_outputs] + decoder_states)

    # Attention model
    attn_decoder_outputs =Input(shape = (None,configuration.latent_dim * 2,), name = "Dec_output_receptor")
    attn_encoder_outputs =Input(shape = (None,configuration.latent_dim * 2,), name = "Enc_output_receptor")

    attention = Dot(axes=[2, 2])([attn_decoder_outputs, attn_encoder_outputs])
    attention = Activation('softmax')(attention)

    context = Dot(axes=[2, 1])([attention,attn_encoder_outputs]) 
    attn_decoder_combined_context = Concatenate(axis = -1)([context, attn_decoder_outputs])

    first_dense = full_model.layers[12]
    final_dense = full_model.layers[13]

    pre_decoder_outputs = first_dense(attn_decoder_combined_context) # equation (5) of the paper
    attn_outputs = final_dense(pre_decoder_outputs)
    attention_model = Model([attn_decoder_outputs,attn_encoder_outputs], attn_outputs)

    return encoder_model, decoder_model, attention_model

def predict_bidirectional_dot_attention_seq2seq_single_record (encoder_input_data : Any, raw_score_data: Any, encoder_model : Model, decoder_model: Model, configuration: Configuration, mse_log_full_path: str, array_diff_full_path:str) -> list:
    
    encoder_output, encoder_state_h, encoder_state_c = encoder_model.predict(encoder_input_data)
    result = []
    current_dec_input = 41. # <- Starting Symbol
    current_h = encoder_state_h
    current_c = encoder_state_c
    while True:
        output,current_h,current_c = decoder_model.predict([np.array([[current_dec_input]]),current_h,current_c])
        current_dec_input = np.argmax(output[0,-1,:])
        result.append(current_dec_input)
        if len(result) == raw_score_data.shape[1] - 1:
            break

    return result

def predict_bidirectional_dot_attention_seq2seq_batch (encoder_input_data : Any, raw_score_data: Any, encoder_model : Model, decoder_model: Model, attention_model: Model, configuration: Configuration, mse_log_full_path: str, array_diff_full_path:str):
    # array_diff_file = open(array_diff_full_path, 'w')
    # mse_log_file = open(mse_log_full_path, 'w')

    inf_enc_out, inf_enc_state_h, inf_enc_state_c = encoder_model.predict(encoder_input_data)
    results = np.array([])
    current_dec_input = np.ones((encoder_input_data.shape[0], 1)) * 41.
    current_h = inf_enc_state_h
    current_c = inf_enc_state_c
    while True:
        pre_output,current_h,current_c = decoder_model.predict([current_dec_input,current_h,current_c])
        output = attention_model.predict([pre_output,inf_enc_out])
        current_dec_input = output.argmax(axis=-1)
        if results.shape[0] == 0:
            results = current_dec_input
        else:
            results = np.concatenate((results,current_dec_input),axis = -1)
        if results.shape[-1] == (encoder_input_data.shape[-1]):
            break
    
    # array_diff_file.close()
    # mse_log_file.close()

    return results

def main(args) :
    feature_file = args[1]
    training_hist_path = args[2]
    model_path = args[3]
    predicted_diff_path = args[4]
    mse_progress = args[5]
    experiment_name = args[6]

    print("Experiment Name", experiment_name)

    model_full_path = model_path + '/' + experiment_name + '.h5'
    encoder_model_full_path = model_path + '/' + experiment_name + '_encoder.h5'
    decoder_model_full_path = model_path + '/' + experiment_name + '_decoder.h5'
    attention_model_full_path = model_path + '/' + experiment_name + '_attention.h5'

    array_diff_full_file_name = predicted_diff_path + '/' + experiment_name + '.diff'
    mse_log_full_file_name = mse_progress + '/' + experiment_name + '_MSE.csv'

    configuration = Configuration(
        experiment_name = experiment_name,
        latent_dim=32,
        num_encoder_tokens = 5,
        num_decoder_tokens = 42,
        num_encoder_embed = 2,
        num_decoder_embed = 32,
        seq_num= 10000,
        seq_len = 90,
        base_learning_rate=0.001,
        batch_size=int(10000 * 0.01),
        loss='sparse_categorical_crossentropy'
    )

    # Load Dataset
    encoder_input_data, decoder_input_data, decoder_target_data, raw_score_data = load_seq2_seq_data(feature_file, configuration)

    # Build Full Model
    full_model = build_bidirectional_dot_attention_seq2seq_model(configuration, model_full_path, training_hist_path, encoder_input_data, decoder_input_data, decoder_target_data)
    full_model.save(model_full_path)

    # Transform full model to attention, encoder and decoder model
    # encoder_model, decoder_model = transform_model('Results/model_experiment/model/seq2seq/Seq2Seq_Bidirectional_L32_E32_Lr0-001_BSm00-1_10000.h5', configuration)
    encoder_model, decoder_model = transform_model(full_model , configuration)
    encoder_model.save(encoder_model_full_path)
    decoder_model.save(decoder_model_full_path)

    # Predict data
    pred = predict_bidirectional_dot_attention_seq2seq_batch(encoder_input_data, raw_score_data, encoder_model, decoder_model, configuration, mse_log_full_file_name, array_diff_full_file_name)
    
    # FIXME properly save predicted data
    pred_file = open('pred.pickle', 'wb')
    pickle.dump(pred, pred_file)