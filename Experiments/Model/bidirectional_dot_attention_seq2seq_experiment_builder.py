import sys
import os
import time
import math
import pandas as pd
import numpy as np

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention, TimeDistributed, Dot, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from Seq2SeqExperimentInterface import Seq2SeqExperimentInterface
from typing import Any
from utilities import load_seq2_seq_data, generate_training_statistic_file
from Configuration import Configuration
from CustomCallbacks import WarmUpReduceLROnPlateau

# Bidirectional LSTM with Dot Attention Seq2Seq Model Experiment
# INPUT: Feature File Path
# OUTPUT: History File, Model File, Array Diff Result

class BidirectionalDotAttentionSeq2SeqExperimentBuilder (Seq2SeqExperimentInterface) :
    def __init__ (self, configuration : Configuration, feature_file_path: str, 
    training_hist_base_path : str = 'Results/model_experiment/training_stat/seq2seq', 
    model_base_path: str = 'Results/model_experiment/model/seq2seq', 
    array_diff_base_path: str = 'Results/model_experiment/predicted_diff', 
    mse_log_base_path: str = 'Results/model_experiment/mse_log',
    experiment_name_prefix: str = 'Seq2Seq_Bidirectional_DotAttention') -> None:

        super().__init__(configuration, experiment_name_prefix=experiment_name_prefix)

        # Init Config and Experiment Name
        self.__configuration = configuration
        self.__experiment_name_prefix = experiment_name_prefix
        self.__experiment_name = super().get_experiment_name()

        # Path Init
        self.__base_training_hist_path = training_hist_base_path
        self.__training_hist_path = training_hist_base_path + '/' + self.__experiment_name + '.model_hist'
        self.__array_diff_path = array_diff_base_path + '/' + self.__experiment_name + '.diff'
        self.__mse_log_path = mse_log_base_path + '/' + self.__experiment_name + '_MSE.csv'

        self.__full_model_path = model_base_path + '/' + self.__experiment_name + '.h5'
        self.__encoder_model_path = model_base_path + '/' + self.__experiment_name + '_encoder.h5'
        self.__decoder_model_path = model_base_path + '/' + self.__experiment_name + '_decoder.h5'
        self.__attention_model_path = model_base_path + '/' + self.__experiment_name + '_attention.h5'

        # Input Data Init
        self.__feature_file_path = feature_file_path
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None

        print(self.__experiment_name, ' has been created')

    # Utilities Functions
    def __get_real_batch_size (self) -> int :
        # If Batch Size is multiply so input as decimal
        if self.__configuration.batch_size < 1 :
            return int(self.__configuration.batch_size * self.__configuration.seq_num)
        else :
            return self.__configuration.batch_size

    def __write_offset_to_file (self, offset_list) -> None:
        destination_file = open(self.__array_diff_path, 'w')

        for read in offset_list :
            destination_file.write(str(read)[1:-1].replace(' ', '') + '\n')

        destination_file.close()

    def load_full_model (self, full_model_path: str) -> None:
        self.__full_model_path = full_model_path
        self.__full_model = load_model(self.__full_model_path)

    # Sub-Pipeline functions

    def load_data (self) -> None :
        feature_file = open(self.__feature_file_path, 'r')

        encoder_input_data = list()
        decoder_target_data = list()

        line = feature_file.readline()

        current_rec = 0

        while line != '' :
            line_component = line[:-1].split(',')
            no_of_feature = int(len(line_component) / 2)
            encoded_seq = line_component[:no_of_feature]
            raw_score = line_component[no_of_feature:]

            encoder_input_data.append(encoded_seq)
            decoder_target_data.append(raw_score)

            current_rec += 1

            if current_rec == self.__configuration.seq_num :
                break
            
            line = feature_file.readline()

        feature_file.close()

        encoder_input_data = np.array(encoder_input_data,dtype="float32")
        decoder_target_data = np.array(decoder_target_data,dtype="float32")
        decoder_input_data = np.concatenate([np.ones((decoder_target_data.shape[0],1))*41.,decoder_target_data[:,:-1]],axis = -1)

        self.__encoder_input_data = encoder_input_data
        self.__decoder_target_data = decoder_target_data
        self.__decoder_input_data = decoder_input_data

    def build_model (self, keras_verbose=1) -> None :
        # Load the data when there is no data in the obj
        if self.__encoder_input_data is None or self.__decoder_input_data is None or self.__decoder_target_data is None:
            print('Data did not be loaded. Loading now...')
            self.load_data()

        # Encoder Part
        encoder_inputs = Input(shape=(None,))
        encoder_embed = Embedding(output_dim=self.__configuration.num_encoder_embed, input_dim=self.__configuration.num_encoder_tokens)
        encoder = Bidirectional(LSTM(self.__configuration.latent_dim, return_sequences=True, return_state=True))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embed(encoder_inputs))
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Decoder Part
        decoder_inputs = Input(shape=(None, ))
        decoder_embed = Embedding(output_dim=self.__configuration.num_decoder_embed, input_dim=self.__configuration.num_decoder_tokens)   
        decoder_lstm = LSTM(self.__configuration.latent_dim*2, return_sequences=True, return_state=True)
        decoder_outputs,_,_= decoder_lstm(decoder_embed(decoder_inputs), initial_state=encoder_states)

        # Dot Attention Part
        attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
        attention = Activation('softmax')(attention)
        
        context = Dot(axes=[2, 1])([attention,encoder_outputs]) # dot([attention, encoder], axes=[2,1])
        decoder_combined_context = Concatenate(axis = -1)([context, decoder_outputs])

        pre_decoder_outputs = TimeDistributed(Dense(self.__configuration.latent_dim*2, activation="tanh"))(decoder_combined_context) # equation (5) of the paper
        decoder_dense =  TimeDistributed(Dense(self.__configuration.num_decoder_tokens, activation='softmax'))
        decoder_outputs = decoder_dense(pre_decoder_outputs )

        # Plug everything together
        full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile and Build the model
        earlystop_callback = EarlyStopping(monitor='loss', min_delta=0.00001, patience=16)
        model_checkpoint_callback = ModelCheckpoint(filepath=self.__full_model_path, monitor='accuracy', mode='max', save_best_only=True)
        reduce_lr = WarmUpReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0, init_lr = self.__configuration.base_learning_rate, warmup_batches = self.__configuration.count_train * 5, min_delta = 0.001)

        full_model.compile(optimizer=RMSprop(lr=self.__configuration.base_learning_rate), loss=self.__configuration.loss, metrics=["accuracy"])

        training_hist = full_model.fit([self.__encoder_input_data, self.__decoder_input_data], self.__decoder_target_data, batch_size=self.__get_real_batch_size(), epochs=1000, callbacks=[reduce_lr,earlystop_callback,model_checkpoint_callback], verbose=keras_verbose)
        
        # Save training stat to file
        generate_training_statistic_file(training_hist, self.__experiment_name, destination_file_path = self.__base_training_hist_path)

        self.__full_model = full_model

    def transform_model (self) -> None :
        if self.__full_model is None :
            print('The model was not built. Please build full model before calling this function')
            return None
        
        # Encoder Model
        inf_enc_input = self.__full_model.input[0]
        inf_enc_embed = self.__full_model.layers[1]
        inf_encoder = self.__full_model.layers[3]
        inf_enc_out, inf_fh, inf_fc, inf_bh, inf_bc =  inf_encoder(inf_enc_embed(inf_enc_input))
        inf_enc_state_h = Concatenate()([inf_fh, inf_bh])
        inf_enc_state_c = Concatenate()([inf_fc, inf_bc])
        inf_enc_states = [inf_enc_state_h, inf_enc_state_c]

        encoder_model = Model(inf_enc_input,[inf_enc_out]+inf_enc_states)

        # Decoder Model
        decoder_inputs = self.__full_model.input[1]  # input_2
        decoder_embed_inputs = self.__full_model.layers[4](decoder_inputs)

        decoder_state_input_h = Input(shape=(self.__configuration.latent_dim * 2,), name="input_3")
        decoder_state_input_c = Input(shape=(self.__configuration.latent_dim * 2,), name="input_4")

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.__full_model.layers[7]
        pre_decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_embed_inputs , initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]

        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [pre_decoder_outputs] + decoder_states)

        # Attention model
        attn_decoder_outputs =Input(shape = (None,self.__configuration.latent_dim * 2,), name = "Dec_output_receptor")
        attn_encoder_outputs =Input(shape = (None,self.__configuration.latent_dim * 2,), name = "Enc_output_receptor")

        attention = Dot(axes=[2, 2])([attn_decoder_outputs, attn_encoder_outputs])
        attention = Activation('softmax')(attention)

        context = Dot(axes=[2, 1])([attention,attn_encoder_outputs]) 
        attn_decoder_combined_context = Concatenate(axis = -1)([context, attn_decoder_outputs])

        first_dense = self.__full_model.layers[12]
        final_dense = self.__full_model.layers[13]

        pre_decoder_outputs = first_dense(attn_decoder_combined_context) # equation (5) of the paper
        attn_outputs = final_dense(pre_decoder_outputs)
        attention_model = Model([attn_decoder_outputs,attn_encoder_outputs], attn_outputs)

        # Save model to file
        encoder_model.save(self.__encoder_model_path)
        decoder_model.save(self.__decoder_model_path)
        attention_model.save(self.__attention_model_path)

        # Set as obj attr
        self.__encoder_model = encoder_model
        self.__decoder_model = decoder_model
        self.__attention_model = attention_model

    def predict_data (self) -> np.array :
        if self.__encoder_input_data is None :  
            print('Please load data before calling this function')
            return None
        
        if self.__encoder_model is None or self.__decoder_model is None or  self.__attention_model is None :
            print('The model has been transformed yet. I will transform the model for you first!')
            self.transform_model()

        # encoder_input_data shape is (seq_num, seq_len) need to reshape (no_of_round, seq_num/round, seq_len)
        MAX_CHUNK_SIZE = 30000
        no_of_round = math.ceil(self.__encoder_input_data.shape[0] / MAX_CHUNK_SIZE)

        complete_result = np.array([])
        
        for chunk_no in range(no_of_round) :
            encoder_input_data_chunk = self.__encoder_input_data[chunk_no * MAX_CHUNK_SIZE : (chunk_no + 1) * MAX_CHUNK_SIZE, :]

            encoder_output, inf_enc_state_h, inf_enc_state_c = self.__encoder_model.predict(encoder_input_data_chunk)

            results = np.array([])
            current_dec_input = np.ones((encoder_input_data_chunk.shape[0], 1)) * 41.
            current_h = inf_enc_state_h
            current_c = inf_enc_state_c
            while True:
                pre_output,current_h,current_c = self.__decoder_model.predict([current_dec_input,current_h,current_c])
                output =  self.__attention_model.predict([pre_output,encoder_output])
                current_dec_input = output.argmax(axis=-1)
            
                if results.shape[0] == 0:
                    results = current_dec_input
                else:
                    results = np.concatenate((results,current_dec_input),axis = -1)

                if results.shape[-1] == (encoder_input_data_chunk.shape[-1]):

                    if complete_result.shape[0] == 0 :
                        complete_result = results
                    else :
                        complete_result = np.concatenate([complete_result, results], axis=0)
                    break
        
        return results

    def calculate_diff_error (self, pred_np: np.ndarray) -> (list, int, float) :
        mse_log_file = open(self.__mse_log_path, 'w')

        # Pred Shape,Decoder Target Data  (n_read, seq_len)
        seq_num = min(pred_np.shape[0], self.__decoder_target_data.shape[0])
        seq_len = min(pred_np.shape[1], self.__decoder_target_data.shape[1])

        accum_sigma_distance = 0
        accuracy_corrent_answer_counter = 0

        offset_list = list()

        for read_no in range(0, seq_num) :
            current_pred = pred_np[read_no, :]
            current_target = self.__decoder_target_data[read_no, :]
            diff = np.subtract(current_target, current_pred).astype(int)
            offset_list.append(diff.tolist())
            accuracy_corrent_answer_counter += diff.tolist().count(0)
            
            accum_sigma_distance += np.sum(np.power(diff, 2))
            n_of_data = (read_no+1) * seq_len
            mse = (1/n_of_data) * accum_sigma_distance
            mse_log_file.write(str(mse) + '\n')

            accuracy = accuracy_corrent_answer_counter / n_of_data
        
        mse_log_file.close()

        self.__write_offset_to_file(offset_list)

        return offset_list, mse, accuracy
    
    # Run full pipeline and report the result
    def run (self) -> None:
        # Report Configuration
        print('Running ', self.__experiment_name, 'with the following configuration')
        print('Dataset:', self.__configuration.seq_num, 'read(s) from', self.__feature_file_path.split('/')[-1], ',', self.__configuration.seq_len, 'base(s) per read')
        print('latent_dim:', self.__configuration.latent_dim)
        print('num_encoder_tokens:', self.__configuration.num_encoder_tokens)
        print('num_decoder_tokens:', self.__configuration.num_decoder_tokens)
        print('num_encoder_embed:', self.__configuration.num_encoder_embed)
        print('num_decoder_embed:', self.__configuration.num_decoder_embed)
        print('base_learning_rate:', self.__configuration.base_learning_rate)

        if self.__configuration.batch_size < 1 :    
            print('batch_size: multiply with ', self.__configuration.batch_size, '(' , self.__configuration.batch_size * self.__configuration.seq_num, ')')
        else :
            print('batch_size:', self.__configuration.batch_size)

        print('loss:', str(self.__configuration.loss), '\n')
        
        # Load Data
        start_time = time.time()
        self.load_data()
        load_data_time = time.time() - start_time

        # Build Model and keep model fitting slient
        start_time = time.time()
        self.build_model(keras_verbose=0)
        build_model_time = time.time() - start_time

        # Transform Full Model to Attention, Encoder and Decoder model
        start_time = time.time()
        self.transform_model()
        transform_model_time = time.time() - start_time

        # Predict Data
        start_time = time.time()
        pred = self.predict_data()
        prediction_time = time.time() - start_time

        offset_list, mse, accuracy = self.calculate_diff_error(pred)

        # Getting Model Size in MB (convert from byte)
        full_model_size = os.stat(self.__full_model_path).st_size / 10**6
        encoder_model_size = os.stat(self.__encoder_model_path).st_size / 10**6
        decoder_model_size = os.stat(self.__decoder_model_path).st_size / 10**6
        attention_model_size = os.stat(self.__attention_model_path).st_size / 10**6

        # Getting number of epoch and encoder accuracy
        training_hist = pd.read_csv(self.__training_hist_path).iloc[-1,:]
        encoder_epoch = int(training_hist.epoch)
        encoder_accuracy = training_hist.accuracy

        print('\nExperiment Done in ', load_data_time + build_model_time + transform_model_time + prediction_time, 'sec(s)')
        print('Load Data Time', load_data_time, 'sec(s)')
        print('Build Model Time', build_model_time, 'sec(s) (' , build_model_time/encoder_epoch, 'sec/epoch)')
        print('Transform Model Time', transform_model_time)
        print('Prediction Time', prediction_time, 'sec(s) (', prediction_time /  self.__configuration.seq_num, ' sec/read)')

        print('\nEncoder Model')
        print('Epoch:', encoder_epoch)
        print('Accuracy:', encoder_accuracy)
        print('Final Prediction MSE:', mse)
        print('Final Accuracy:', accuracy)

        print('\nModel Sizes')
        print('Full Model Size:', full_model_size, 'MB')
        print('Encoder Model Size:', encoder_model_size, 'MB')
        print('Decoder Model Size:', decoder_model_size, 'MB')
        print('Attention Model Size:', attention_model_size, 'MB')

        print('\nResult File Path')
        print('Training Hist Path:', self.__training_hist_path)
        print('Predicted Offset Path:', self.__array_diff_path)
        print('MSE Log Path:', self.__mse_log_path)
        print('Full Model Path:', self.__full_model_path)
        print('Encoder Model Path:', self.__encoder_model_path)
        print('Decoder Model Path:', self.__decoder_model_path)
        print('Attention Model Path:', self.__attention_model_path)

    def predict_only (self, full_model_path: str) :
        # Report Configuration
        print('Running ', self.__experiment_name, 'with the following configuration')
        print('Dataset:', self.__configuration.seq_num, 'read(s) from', self.__feature_file_path.split('/')[-1], ',', self.__configuration.seq_len, 'base(s) per read')
        print('latent_dim:', self.__configuration.latent_dim)
        print('num_encoder_tokens:', self.__configuration.num_encoder_tokens)
        print('num_decoder_tokens:', self.__configuration.num_decoder_tokens)
        print('num_encoder_embed:', self.__configuration.num_encoder_embed)
        print('num_decoder_embed:', self.__configuration.num_decoder_embed)
        print('base_learning_rate:', self.__configuration.base_learning_rate)

        if self.__configuration.batch_size < 1 :    
            print('batch_size: multiply with ', self.__configuration.batch_size, '(' , self.__configuration.batch_size * self.__configuration.seq_num, ')')
        else :
            print('batch_size:', self.__configuration.batch_size)

        print('loss:', str(self.__configuration.loss), '\n')
        
        # Load Data
        start_time = time.time()
        self.load_data()
        load_data_time = time.time() - start_time
        print('Load data done in', load_data_time, 'sec(s)')

        # Load Model from existing file
        start_time = time.time()
        self.load_full_model(full_model_path)
        model_loading_time = time.time() - start_time
        print('Load model done in', model_loading_time, 'sec(s)')

        # Transform Full Model to Attention, Encoder and Decoder model
        start_time = time.time()
        self.transform_model()
        transform_model_time = time.time() - start_time
        print('transform model done in', transform_model_time)

        # Predict Data
        start_time = time.time()
        pred = self.predict_data()
        prediction_time = time.time() - start_time
        print('Predict done in', prediction_time, 'sec(s) (', prediction_time /  self.__configuration.seq_num, ' sec/read)')

        offset_list, mse, accuracy = self.calculate_diff_error(pred)

        # Getting Model Size in MB (convert from byte)
        full_model_size = os.stat(self.__full_model_path).st_size / 10**6
        encoder_model_size = os.stat(self.__encoder_model_path).st_size / 10**6
        decoder_model_size = os.stat(self.__decoder_model_path).st_size / 10**6
        attention_model_size = os.stat(self.__attention_model_path).st_size / 10**6

        print('\nExperiment Done in ', load_data_time + model_loading_time + transform_model_time + prediction_time, 'sec(s)')

        print('\nEncoder Model')
        print('Final Prediction MSE:', mse)
        print('Final Accuracy:', accuracy)

        print('\nModel Sizes')
        print('Full Model Size:', full_model_size, 'MB')
        print('Encoder Model Size:', encoder_model_size, 'MB')
        print('Decoder Model Size:', decoder_model_size, 'MB')
        print('Attention Model Size:', attention_model_size, 'MB')

        print('\nResult File Path')
        print('Predicted Offset Path:', self.__array_diff_path)
        print('MSE Log Path:', self.__mse_log_path)
        print('Full Model Path:', self.__full_model_path)
        print('Encoder Model Path:', self.__encoder_model_path)
        print('Decoder Model Path:', self.__decoder_model_path)
        print('Attention Model Path:', self.__attention_model_path)
        
def main(args) :
    feature_file = args[1]
    
    # Sample Experiment 
    sample_experiment = BidirectionalDotAttentionSeq2SeqExperimentBuilder (
        feature_file_path = args[1],
        configuration = Configuration(
        latent_dim=256,
        num_encoder_tokens = 5,
        num_decoder_tokens = 42,
        num_encoder_embed = 2,
        num_decoder_embed = 32,
        seq_num= 90000,
        seq_len = 90,
        base_learning_rate=0.001,
        batch_size=0.001,
        loss='sparse_categorical_crossentropy'
        )
    )

    # Easier method -> run whole pipeline
    # sample_experiment.run()

    # Got the full model ? -> Predict only option
    # sample_experiment.predict_only(<Model Path>)
    sample_experiment.predict_only('Results/model_experiment/model/seq2seq/Seq2Seq_Bidirectional_DotAttention_L256_E32_Lr0-001_BSm0-001_90000.h5')
    
    # Loading Dataset
    # sample_experiment.load_data()

    # Build Full Model
    # sample_experiment.build_model()

    # If full model is existed -> you can load the full model instead via
    # sample_experiment.load_full_model(<Full Model Path>)

    # Transform Full Model to Attention, Encoder and Decoder model
    # sample_experiment.transform_model()

    # Predict Data
    # pred = sample_experiment.predict_data()

    # Calculate offset between actual and predicted data and mse log -> write to file
    # offset_list, mse, accuracy = sample_experiment.calculate_diff_error(pred)
    
if __name__ == "__main__":
    main(sys.argv)