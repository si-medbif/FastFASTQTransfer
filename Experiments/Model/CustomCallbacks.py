from tensorflow.keras import backend as K

class WarmUpReduceLROnPlateau(ReduceLROnPlateau):
  def __init__(self,
               monitor,
               factor,
               patience,
               init_lr,
               min_lr,
               warmup_batches,
               min_delta):
    super().__init__(monitor=monitor,factor=factor,patience=patience,min_lr=min_lr,min_delta=min_delta)
    self.warmup_batches = warmup_batches
    self.init_lr = init_lr
    self.batch_count = 0

  def on_train_batch_begin(self, batch, logs=None):
    if self.batch_count <= self.warmup_batches:
      lr = self.batch_count*self.init_lr/self.warmup_batches
      K.set_value(self.model.optimizer.lr, lr)

  def on_train_batch_end(self, batch, logs=None):
    self.batch_count = self.batch_count + 1