import math

# Configuration Class
# Store the model configuration for each experiment

class Configuration :
    def __init__ (self, experiment_name, latent_dim=256, num_encoder_tokens = 5, num_decoder_tokens = 41, num_encoder_embed = 2, num_decoder_embed = 32, seq_num= 10000, seq_len = 90, base_learning_rate=0.01, batch_size=10, loss='categorical_crossentropy') :
        self.experiment_name = experiment_name
        self.latent_dim = latent_dim
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.num_encoder_embed = num_encoder_embed
        self.num_decoder_embed = num_decoder_embed
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.base_learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.loss = loss
        self.count_train = math.ceil(self.seq_num / self.batch_size)