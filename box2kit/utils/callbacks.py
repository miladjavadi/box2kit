from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class DelayedEarlyStopping(EarlyStopping):
    """
    Early-stopping callback with delayed initialization.
    """
    def __init__(self, warmup_length=0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_length = warmup_length
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_length:
            return
        return super().on_validation_epoch_end(trainer, pl_module)