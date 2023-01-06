from src.models.bert import *
from src.models.NaiveBayes import *

def get_model(config: dict)->ModelInterface:
    model_name = config['model']['name'].lower()
    if model_name == 'bert':
        return BertModel(config)
    if model_name == 'naive_bayes':
        return NBModel(config)
    # add new models here
    # if
    #   ...
    else:
        raise AttributeError(f"found no model class for {config['model']['name']}")