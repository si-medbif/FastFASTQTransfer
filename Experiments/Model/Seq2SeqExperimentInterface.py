from tensorflow.keras.models import load_model

class Seq2SeqExperimentInterface :

    __experiment_name_prefix: str = 'Seq2_Seq_Untitled'
    __experiment_name: str = ''
    __base_training_hist_path: str = ''
    __training_hist_path: str = ''
    __array_diff_path: str = ''
    __mse_log_path: str = ''

    __full_model_path: str = ''
    __encoder_model_path: str = ''
    __decoder_model_path: str = ''
    __attention_model_path: str = ''


    # Model Obj
    self.__full_model = None
    self.__encoder_model = None
    self.__decoder_model = None
    self.__attention_model = None  

    # Name and Path Generator (Need static getter <- Don't want modified path)
    def __generate_experiment_name (self) -> str :
        if self.__configuration.batch_size < 1 :
            batch_size_in_name = 'm' + str(self.__configuration.batch_size).replace('.', '-')
        else :
            batch_size_in_name = str(self.__configuration.batch_size)

        return self.__experiment_name_prefix + '_L' + str(self.__configuration.latent_dim) + '_E' + str(self.__configuration.num_decoder_embed) + '_Lr' + str(self.__configuration.base_learning_rate).replace('.', '-') + '_BS' + batch_size_in_name + '_' + str(self.__configuration.seq_num)

    # Config Getter
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

    @abstractmethod
    def load_data (self) -> None :
        pass

    @abstrctmethod
    def build_model (self, keras_verbose=1) -> None :
        pass

    @abstrctmethod
    def transform_model (self) -> None :
        pass

    @abstractmethod
    def predict_data (self) -> np.array :
        pass
    
    @abstractmethod
    def calculate_diff_error (self, pred_np: np.ndarray) -> (list, int) :
        pass

    @abstractmethod
    def run (self) -> None:
        pass

    @abstractmethod
    def predict_only (self, full_model_path: str) :
        pass