import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from Configuration import Configuration

class Seq2SeqExperimentInterface :

    # Model Obj
    __full_model = None
    __encoder_model = None
    __decoder_model = None
    __attention_model = None  

    def __init__ (self, configuration : Configuration, 
        training_hist_base_path : str = 'Results/model_experiment/training_stat/seq2seq', 
        model_base_path: str = 'Results/model_experiment/model/seq2seq', 
        array_diff_base_path: str = 'Results/model_experiment/predicted_diff', 
        mse_log_base_path: str = 'Results/model_experiment/mse_log',
        experiment_name_prefix: str = 'Seq2_Seq_Untitled') -> None:

        self.__experiment_name_prefix = experiment_name_prefix
        self.__configuration = configuration
        self.__experiment_name = self.__generate_experiment_name()

        # Path Init
        self.__base_training_hist_path = training_hist_base_path
        self.__training_hist_path = training_hist_base_path + '/' + self.__experiment_name + '.model_hist'
        self.__array_diff_path = array_diff_base_path + '/' + self.__experiment_name + '.diff'
        self.__mse_log_path = mse_log_base_path + '/' + self.__experiment_name + '_MSE.csv'

        self.__full_model_path = model_base_path + '/' + self.__experiment_name + '.h5'
        self.__encoder_model_path = model_base_path + '/' + self.__experiment_name + '_encoder.h5'
        self.__decoder_model_path = model_base_path + '/' + self.__experiment_name + '_decoder.h5'
        self.__attention_model_path = model_base_path + '/' + self.__experiment_name + '_attention.h5'

    # Name and Path Generator (Need static getter <- Don't want modified path)
    def __generate_experiment_name (self) -> str :
        print(self.__configuration.latent_dim)
        if self.__configuration.batch_size < 1 :
            batch_size_in_name = 'm' + str(self.__configuration.batch_size).replace('.', '-')
        else :
            batch_size_in_name = str(self.__configuration.batch_size)

        return self.__experiment_name_prefix + '_L' + str(self.__configuration.latent_dim) + '_E' + str(self.__configuration.num_decoder_embed) + '_Lr' + str(self.__configuration.base_learning_rate).replace('.', '-') + '_BS' + batch_size_in_name + '_' + str(self.__configuration.seq_num)

    # Getter Function
    def get_training_hist_path (self) -> str:
        return self.__training_hist_path
    
    def get_array_diff_path (self) -> str:
        return self.__array_diff_path
    
    def get_mse_log_path (self) -> str:
        return self.__mse_log_path
    
    def get_full_model_path (self) -> str:
        return self.__full_model_path
    
    def get_encoder_model_path (self) -> str:
        return self.__encoder_model_path
    
    def get_decoder_model_path (self) -> str:
        return self.__decoder_model_path
    
    def get_attention_model_path (self) -> str:
        return self.__attention_model_path

    def get_experiment_name (self) -> str :
        return self.__experiment_name

    def get_training_hist_path (self) -> str:
        return self.__training_hist_path
    
    def get_array_diff_path (self) -> str :
        return self.__array_diff_path
    
    def get_mse_progress_path (self) -> str :
        return self.__mse_log_path

    # Model Setter
    def load_full_model (self, full_model_path: str) -> None:
        self.__full_model_path = full_model_path
        self.__full_model = load_model(self.__full_model_path)

    # Model Getters
    def get_full_model (self) -> Model:
        return self.__full_model
    
    def get_encoder_model (self) -> Model:
        return self.__encoder_model
    
    def get_decoder_model (self) -> Model:
        return self.__decoder_model
    
    def get_attention_model (self) -> Model:
        return self.__attention_model

    # Abstract Methods

    def load_data (self) -> None :
        pass

    def build_model (self, keras_verbose=1) -> None :
        pass

    def transform_model (self) -> None :
        pass

    def predict_data (self) -> np.array :
        pass
    
    def calculate_diff_error (self, pred_np: np.ndarray) -> (list, int) :
        pass

    def run (self) -> None:
        pass

    def predict_only (self, full_model_path: str) :
        pass