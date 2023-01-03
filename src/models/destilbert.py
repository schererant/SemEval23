

class DestilBert:
    def __init__(self,
                 model_name: str = 'distilbert-base-uncased',
                 max_length: int = 512,
                 batch_size: int = 32,
                 num_labels: int = 2,
                 num_epochs: int = 3,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 0,
                 gradient_accumulation_steps: int = 1,
                 seed: int = 42,
                 device: str = 'cpu',
                 verbose: bool = False,
                 **kwargs):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.kwargs = kwargs