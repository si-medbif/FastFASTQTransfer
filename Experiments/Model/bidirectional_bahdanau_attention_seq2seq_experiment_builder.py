import sys 
import math
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow import TensorShape
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention, TimeDistributed, Activation, Layer, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from Seq2SeqExperimentInterface import Seq2SeqExperimentInterface
from utilities import generate_training_statistic_file
from Configuration import Configuration
from CustomCallbacks import WarmUpReduceLROnPlateau

# Bidirectional LSTM with Bahdanau Attention Seq2Seq Model Experiment
# INPUT: Feature File Path
# OUTPUT: History File, Model File, Array Diff Result

class BidirectionalBahdanauAttentionSeq2SeqExperimentBuilder (Seq2SeqExperimentInterface) :
    def __init__ (self, configuration : Configuration, feature_file_path: str, 
        training_hist_base_path : str = 'Results/model_experiment/training_stat/seq2seq', 
        model_base_path: str = 'Results/model_experiment/model/seq2seq', 
        array_diff_base_path: str = 'Results/model_experiment/predicted_diff', 
        mse_log_base_path: str = 'Results/model_experiment/mse_log',
        experiment_name_prefix: str = 'Seq2Seq_Bidirectional_BahdanauAttention') -> None:

            # Init Config and Experiment Name
            self.__configuration = configuration
            super().__init__(self.__configuration, experiment_name_prefix=experiment_name_prefix)
            self.__experiment_name = super().get_experiment_name()

            # Input Data Init
            self.__feature_file_path = feature_file_path
            self.__encoder_input_data = None
            self.__decoder_input_data = None
            self.__decoder_target_data = None

            print(self.__experiment_name, ' has been created')

    # Utilities Functions
    def __get_real_batch_size (self) -> int :
        # If Batch Size is multiply so input as decimal
        if self.__configuration.batch_size < 1 :
            return int(self.__configuration.batch_size * self.__configuration.seq_num)
        else :
            return self.__configuration.batch_size

    def __write_offset_to_file (self, offset:list) -> None :
        offset_file = open(super().get_array_diff_path(),'w')

        for line in offset :
            offset_file.write(str(line).replace('[','').replace(']','') + '\n')
        
        offset_file.close()

    def __report_configuration (self) -> None :
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

    def load_data (self) -> None :
        feature_file = open(self.__feature_file_path, 'r')

        encoder_input_data = list()
        decoder_input_data = list()
        decoder_target_data = list()

        current_rec = 0
        line = feature_file.readline()

        while line != '' :
            line_component = line[:-1].split(',')
            no_of_feature = int(len(line_component) / 2)
            encoded_seq = line_component[:no_of_feature]
            raw_score = line_component[no_of_feature:]

            encoder_input_data.append(encoded_seq)
            decoder_target_data.append([41] + raw_score)

            current_rec += 1

            if current_rec == self.__configuration.seq_num :
                break
            
            line = feature_file.readline()

        encoder_input_data = np.array(encoder_input_data,dtype="float32")
        decoder_target_data = np.array(decoder_target_data,dtype="float32")
        decoder_input_data = np.concatenate([np.ones((decoder_target_data.shape[0],1))*41.,decoder_target_data[:,:-1]],axis = -1)

        self.__encoder_input_data = encoder_input_data
        self.__decoder_input_data = decoder_input_data
        self.__decoder_target_data = decoder_target_data

    def build_model (self, keras_verbose=1) -> None :

        # Encoder
        encoder_inputs = Input(shape=(None,))
        encoder_embed = Embedding(output_dim=self.__configuration.num_encoder_embed, input_dim=self.__configuration.num_encoder_tokens)
        encoder = Bidirectional(LSTM(self.__configuration.latent_dim, return_sequences=True, return_state=True))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embed(encoder_inputs))
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        #Attention Layer (Bahdanau)
        W1 = Dense(self.__configuration.latent_dim * 2)
        W2 = Dense(self.__configuration.latent_dim * 2)
        V = Dense(1)

        query_with_time_axis = tf.expand_dims(state_h, 1)
        score = V(tf.nn.tanh(W1(query_with_time_axis) + W2(encoder_outputs)))
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # Decoder
        decoder_inputs = Input(shape=(None, ))
        decoder_embed = Embedding(output_dim=self.__configuration.num_decoder_embed, input_dim=self.__configuration.num_decoder_tokens)   
        decoder_lstm = LSTM(self.__configuration.latent_dim*2, return_sequences=True, return_state=True)

        all_outputs = []
        for i in range(self.__configuration.seq_len):
            decoder_embed_concat = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(decoder_embed(decoder_inputs[:,i]),1)], axis=-1)
            single_decoder_outputs,dec_h,dec_c= decoder_lstm(decoder_embed_concat, initial_state=encoder_states)
            encoder_states = [dec_h,dec_c]

            all_outputs.append(single_decoder_outputs)

        all_outputs.append(single_decoder_outputs)

        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        decoder_dense = TimeDistributed(Dense(self.__configuration.num_decoder_tokens, activation="softmax"))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Force save model to file
        model_json = model.to_json()
        model_file = open(super().get_full_model_path() + '.json', 'w')
        model_file.write(model_json)
        model_file.close()

        reduce_lr = WarmUpReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0, init_lr = self.__configuration.base_learning_rate, warmup_batches = self.__configuration.count_train * 5, min_delta = 0.001)
        earlystop_callback = EarlyStopping(monitor='loss', min_delta=0.00001, patience=16)
        model_checkpoint_callback = ModelCheckpoint(filepath=super().get_full_model_path(), monitor='accuracy', mode='max', save_weights_only= True, save_best_only=True)

        # Compile and fit the model
        model.compile(optimizer=RMSprop(lr=self.__configuration.base_learning_rate), loss=self.__configuration.loss, metrics=["accuracy"])

        # Fit the model
        print(self.__encoder_input_data.shape)
        print(self.__decoder_input_data.shape)
        print(self.__decoder_target_data.shape)

        training_hist = model.fit(
            [self.__encoder_input_data, self.__decoder_input_data],
            self.__decoder_target_data,
            batch_size=self.__get_real_batch_size(),
            epochs=1000,
            callbacks=[reduce_lr,earlystop_callback,model_checkpoint_callback]
        )

        # Save training stat to file
        generate_training_statistic_file(training_hist, self.__experiment_name, destination_file_path = super().get_training_hist_path())

        self.__full_model = model

    def transform_model (self) -> None :
        if self.__full_model is None :
            print('The model was not built. Please build full model before calling this function')
            return None

        # Encoder Model
        inf_encoder_inputs = Input(shape=(None,))
        inf_encoder_embed = self.__configuration.layers[1]
        inf_encoder = self.__configuration.layers[2]
        inf_encoder_outputs, forward_h, forward_c, backward_h, backward_c = inf_encoder(inf_encoder_embed(inf_encoder_inputs))
        inf_state_h = Concatenate()([forward_h, backward_h])
        inf_state_c = Concatenate()([forward_c, backward_c])
        inf_encoder_states = [inf_state_h, inf_state_c]

        #Attention Layer
        inf_W1 = self.__full_model[5]
        inf_W2 = self.__full_model[6]
        inf_V = self.__full_model[9]

        inf_query_with_time_axis = tf.expand_dims(inf_state_h, 1)
        inf_score = inf_V(tf.nn.tanh(inf_W1(inf_query_with_time_axis) + inf_W2(inf_encoder_outputs)))
        inf_attention_weights = tf.nn.softmax(inf_score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = inf_attention_weights * inf_encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # Decoder
        inf_decoder_inputs = Input(shape=(1, ))
        inf_decoder_embed = self.__full_model.layers[15] #keras.layers.Embedding(output_dim=num_decoder_embed, input_dim=num_decoder_tokens)   
        inf_decoder_lstm = self.__full_model.layers[25] #keras.layers.LSTM(latent_dim*2, return_sequences=True, return_state=True)

        K_argmax = Lambda(lambda x: K.argmax(x, axis=-1))

        final_dense = self.__full_model.layers[-1]

        encoder_states = inf_encoder_states
        single_dec_input = inf_decoder_inputs

        for i in range(self.__configuration.seq_len):
            inf_decoder_embed_concat = tf.concat([tf.expand_dims(context_vector, 1), inf_decoder_embed(single_dec_input)], axis=-1)
            inf_single_decoder_outputs,dec_h,dec_c= inf_decoder_lstm(inf_decoder_embed_concat, initial_state=encoder_states)
            encoder_states = [dec_h,dec_c]

            inf_single_decoder_outputs = final_dense(inf_single_decoder_outputs)
            single_dec_input = K_argmax(inf_single_decoder_outputs)
            if i == 0:
                all_outputs = single_dec_input
            else:
                all_outputs = tf.concat([all_outputs ,single_dec_input], axis = -1)

        inf_model = Model([inf_encoder_inputs, inf_decoder_inputs], all_outputs)
        
        self.__decoder_model = inf_model

        # Save models to file
        inf_model.save(super().get_decoder_model_path())

    def predict_data (self) -> np.array :
        if self.__encoder_input_data is None :  
            print('Please load data before calling this function')
            return None
        
        if self.__decoder_model is None :
            print('The model has been transformed yet. I will transform the model for you first!')
            self.transform_model()
        
        # encoder_input_data shape is (seq_num, seq_len) need to reshape (no_of_round, seq_num/round, seq_len)
        PRED_BATCH_SIZE = 100
        no_of_round = math.ceil(self.__encoder_input_data.shape[0] / PRED_BATCH_SIZE)

        for i in range(no_of_round):
            result = self.__decoder_model.predict([self.__encoder_input_data[(i*PRED_BATCH_SIZE):((i+1)*PRED_BATCH_SIZE)], self.__decoder_input_data[(i*PRED_BATCH_SIZE):((i+1)*PRED_BATCH_SIZE),0]])
            if i == 0:
                # Init list of result
                results = result
            else:
                results = np.concatenate([results,result], axis=0)
                        
        return results
    
    def calculate_diff_error (self, pred_np: np.ndarray) -> (list, int, float) :
        mse_log_file = open(super().get_mse_log_path(), 'w')

        # Pred Shape,Decoder Target Data  (n_read, seq_len)
        seq_num = min(pred_np.shape[0], self.__decoder_target_data.shape[0])
        seq_len = min(pred_np.shape[1], self.__decoder_target_data.shape[1])

        accum_sigma_distance = 0

        no_of_base = 0
        no_of_correct_predicted = 0

        offset_list = list()

        for read_no in range(0, seq_num) :
            current_pred = pred_np[read_no, :]
            current_target = self.__decoder_target_data[read_no, :]
            diff = np.subtract(current_target, current_pred).astype(int)

            offset_list.append(diff.tolist())

            no_of_base += len(diff.tolist())
            no_of_correct_predicted += diff.tolist().count(0)
            
            accum_sigma_distance += np.sum(np.power(diff, 2))
            n_of_data = (read_no+1) * seq_len
            mse = (1/n_of_data) * accum_sigma_distance
            mse_log_file.write(str(mse) + '\n')
        
        mse_log_file.close()

        self.__write_offset_to_file(offset_list)

        return offset_list, mse, no_of_correct_predicted/no_of_base

    def run (self) -> None:
        # Report Configuration
        self.__report_configuration()
        
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

        offset_list, mse, decoder_accuracy = self.calculate_diff_error(pred)

        # Getting Model Size in MB (convert from byte)
        full_model_size = os.stat(super().get_full_model_path()).st_size / 10**6
        decoder_model_size = os.stat(super().get_decoder_model_path()).st_size / 10**6

        # Getting number of epoch and encoder accuracy
        training_hist = pd.read_csv(super().get_training_hist_path()).iloc[-1,:]
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
        print('Final Prediction Accuracy:', decoder_accuracy)

        print('\nModel Sizes')
        print('Full Model Size:', full_model_size, 'MB')
        print('Inference Model Size:', decoder_model_size, 'MB')

        print('\nResult File Path')
        print('Training Hist Path:', super().get_training_hist_path())
        print('Predicted Offset Path:', super().get_array_diff_path())
        print('MSE Log Path:', super().get_mse_log_path())
        print('Full Model Path:', super().get_full_model_path())
        print('Inference Model Path:', super().get_decoder_model_path())

    def train_only (self) -> None :
        # Report Configuration
        self.__report_configuration()

        # Load Data
        start_time = time.time()
        self.load_data()
        load_data_time = time.time() - start_time
        print('Load data done in', load_data_time, 'sec(s)')

        # Build Model and keep model fitting slient
        start_time = time.time()
        self.build_model(keras_verbose=0)
        build_model_time = time.time() - start_time

        # Getting Model Size in MB (convert from byte)
        full_model_size = os.stat(super().get_full_model_path()).st_size / 10**6

        # Getting number of epoch and encoder accuracy
        training_hist = pd.read_csv(super().get_training_hist_path()).iloc[-1,:]
        encoder_epoch = int(training_hist.epoch)
        encoder_accuracy = training_hist.accuracy

        print('\n' + super().get_experiment_name())
        print('Experiment Done in ', load_data_time + build_model_time, 'sec(s)')
        print('Load Data Time', load_data_time, 'sec(s)')
        print('Build Model Time', build_model_time, 'sec(s) (' , build_model_time/encoder_epoch, 'sec/epoch)')
        print('Accuracy:', encoder_accuracy)

        print('\nModel Sizes')
        print('Full Model Size:', full_model_size, 'MB')
        

    def predict_only (self, full_model_path: str) -> None:
        # Report Configuration
        self.__report_configuration()

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
        full_model_size = os.stat(super().get_full_model_path()).st_size / 10**6
        decoder_model_size = os.stat(super().get_decoder_model_path()).st_size / 10**6

        print('\nExperiment Done in ', load_data_time + model_loading_time + transform_model_time + prediction_time, 'sec(s)')

        print('\nEncoder Model')
        print('Final Prediction MSE:', mse)
        print('Final Accuracy:', accuracy)

        print('\nModel Sizes')
        print('Full Model Size:', full_model_size, 'MB')
        print('Inference Model Size:', decoder_model_size, 'MB')

        print('\nResult File Path')
        print('Predicted Offset Path:', self.__array_diff_path)
        print('MSE Log Path:', self.__mse_log_path)
        print('Full Model Path:', self.__full_model_path)
        print('Inference Model Path:', self.__decoder_model_path)

def main(args) :
    feature_file_path = args[1]
    
    # Sample Experiment 
    sample_experiment = BidirectionalBahdanauAttentionSeq2SeqExperimentBuilder (
        feature_file_path = feature_file_path,
        experiment_name_prefix= 'Seq2Seq_Bidirectional_BahdanauAttention_Tester',
        configuration = Configuration(
        latent_dim=128,
        num_encoder_tokens = 5,
        num_decoder_tokens = 42,
        num_encoder_embed = 2,
        num_decoder_embed = 32,
        seq_num= 1000,
        seq_len = 90,
        base_learning_rate=0.001,
        batch_size=100,
        loss='sparse_categorical_crossentropy'
        )
    )

    # Easier method -> run whole pipeline
    sample_experiment.run()

    # Just Fit the model
    # sample_experiment.train_only()

    # Got the full model ? -> Predict only option
    # sample_experiment.predict_only(<Model Path>)
    # sample_experiment.predict_only('Results/model_experiment/model/seq2seq/Seq2Seq_Bidirectional_DotAttention_L128_E1024_Lr0-001_BSm0-01_10000.h5')
    
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
    # offset_list, mse = sample_experiment.calculate_diff_error(pred)
    
if __name__ == "__main__":
    main(sys.argv)