import sys 
import math
import time
import os
import pandas as pd
import numpy as np

from tensorflow.keras import backend as K
from tensorflow import TensorShape
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention, TimeDistributed, Dot, Activation, Layer
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
            decoder_target_data.append(raw_score)

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

        # Decoder
        decoder_inputs = Input(shape=(None, ))
        decoder_embed = Embedding(output_dim=self.__configuration.num_decoder_embed, input_dim=self.__configuration.num_decoder_tokens)   
        decoder_lstm = LSTM(self.__configuration.latent_dim*2, return_sequences=True, return_state=True)
        decoder_outputs,_,_= decoder_lstm(decoder_embed(decoder_inputs), initial_state=encoder_states)

        #Attention Layer
        attn_layer = BahdanauAttentionLayer(name='attention_layer') 
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention output and decoder LSTM output 
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        decoder_dense = TimeDistributed(Dense(self.__configuration.num_decoder_tokens, activation="softmax"))
        decoder_outputs = decoder_dense(decoder_concat_input)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        reduce_lr = WarmUpReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0, init_lr = self.__configuration.base_learning_rate, warmup_batches = self.__configuration.count_train * 5, min_delta = 0.001)
        earlystop_callback = EarlyStopping(monitor='loss', min_delta=0.00001, patience=16)
        model_checkpoint_callback = ModelCheckpoint(filepath=super().get_full_model_path(), monitor='accuracy', mode='max', save_weights_only= True, save_best_only=True)

        # Compile and fit the model
        model.compile(optimizer=RMSprop(lr=self.__configuration.base_learning_rate), loss=self.__configuration.loss, metrics=["accuracy"])
    
        training_hist = model.fit(
            [self.__encoder_input_data, self.__decoder_input_data],
            self.__decoder_target_data,
            batch_size=self.__get_real_batch_size(),
            epochs=1000,
            callbacks=[reduce_lr,earlystop_callback,model_checkpoint_callback]
        )

        # Save training stat to file
        generate_training_statistic_file(training_hist, self.__experiment_name, destination_file_path = super().get_training_hist_path())

        # Force save model to file
        model.save(super().get_full_model_path())

        self.__full_model = model

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
        pre_decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_embed_inputs , initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]

        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [pre_decoder_outputs] + decoder_states
        )

        # Attention Model
        attn_decoder_outputs =Input(shape = (None,self.__configuration.latent_dim * 2,), name = "Dec_output_receptor")
        attn_encoder_outputs =Input(shape = (None,self.__configuration.latent_dim * 2,), name = "Enc_output_receptor")
        attn_layer = self.__full_model.layers[8]
        attn_out, attn_states = attn_layer([attn_encoder_outputs, attn_decoder_outputs])

        # Concat attention output and decoder LSTM output 
        attn_decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([attn_decoder_outputs, attn_out])

        attn_decoder_dense = self.__full_model.layers[10]
        final_decoder_outputs = attn_decoder_dense(attn_decoder_concat_input)

        attention_model = Model([attn_encoder_outputs, attn_decoder_outputs], final_decoder_outputs)

        self.__encoder_model = encoder_model
        self.__decoder_model = decoder_model
        self.__attention_model = attention_model

        # Save models to file
        encoder_model.save(super().get_encoder_model_path())
        decoder_model.save(super().get_decoder_model_path())
        attention_model.save(super().get_attention_model_path())

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
    
    def calculate_diff_error (self, pred_np: np.ndarray) -> (list, int) :
        mse_log_file = open(super().get_mse_log_path(), 'w')

        # Pred Shape,Decoder Target Data  (n_read, seq_len)
        seq_num = min(pred_np.shape[0], self.__decoder_target_data.shape[0])
        seq_len = min(pred_np.shape[1], self.__decoder_target_data.shape[1])

        accum_sigma_distance = 0

        offset_list = list()

        for read_no in range(0, seq_num) :
            current_pred = pred_np[read_no, :]
            current_target = self.__decoder_target_data[read_no, :]
            diff = np.subtract(current_target, current_pred).astype(int)

            offset_list.append(diff.tolist())
            
            accum_sigma_distance += np.sum(np.power(diff, 2))
            n_of_data = (read_no+1) * seq_len
            mse = (1/n_of_data) * accum_sigma_distance
            mse_log_file.write(str(mse) + '\n')
        
        mse_log_file.close()

        self.__write_offset_to_file(offset_list)

        return offset_list, mse

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

        offset_list, mse = self.calculate_diff_error(pred)

        # Getting Model Size in MB (convert from byte)
        full_model_size = os.stat(super().get_full_model_path()).st_size / 10**6
        encoder_model_size = os.stat(super().get_encoder_model_path()).st_size / 10**6
        decoder_model_size = os.stat(super().get_decoder_model_path()).st_size / 10**6
        attention_model_size = os.stat(super().get_attention_model_path()).st_size / 10**6

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

        print('\nModel Sizes')
        print('Full Model Size:', full_model_size, 'MB')
        print('Encoder Model Size:', encoder_model_size, 'MB')
        print('Decoder Model Size:', decoder_model_size, 'MB')
        print('Attention Model Size:', attention_model_size, 'MB')

        print('\nResult File Path')
        print('Training Hist Path:', super().get_training_hist_path())
        print('Predicted Offset Path:', super().get_array_diff_path())
        print('MSE Log Path:', super().get_mse_log_path())
        print('Full Model Path:', super().get_full_model_path())
        print('Encoder Model Path:', super().get_encoder_model_path())
        print('Decoder Model Path:', super().get_decoder_model_path())
        print('Attention Model Path:', super().get_attention_model_path())

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
        encoder_model_size = os.stat(super().get_encoder_model_path()).st_size / 10**6
        decoder_model_size = os.stat(super().get_decoder_model_path()).st_size / 10**6
        attention_model_size = os.stat(super().get_attention_model_path()).st_size / 10**6

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

class BahdanauAttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(BahdanauAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(BahdanauAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

def main(args) :
    feature_file_path = args[1]
    
    # Sample Experiment 
    sample_experiment = BidirectionalBahdanauAttentionSeq2SeqExperimentBuilder (
        feature_file_path = feature_file_path,
        configuration = Configuration(
        latent_dim=128,
        num_encoder_tokens = 5,
        num_decoder_tokens = 42,
        num_encoder_embed = 2,
        num_decoder_embed = 32,
        seq_num= 10000,
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