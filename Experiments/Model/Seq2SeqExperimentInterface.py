class Seq2SeqExperimentInterface :
    
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