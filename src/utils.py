import yaml
import os
from transformers import AutoModelForSequenceClassification
import torch


# Load yaml config file
def load_config(config_name):

    with open(os.path.join('config/', config_name), 'r') as file:
        config = yaml.safe_load(file)

    return config


def load_model_from_data_dir(model_dir, num_labels):
    """Loads Bert model from specified directory and converts to CUDA model if available"""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    if torch.cuda.is_available():
        return model.to('cuda')
    return model


def get_opt_th(scores, eval_metric):
    max_score = max(s[eval_metric[0]][eval_metric[1]] for s in scores)
    max_index = [s[eval_metric[0]][eval_metric[1]] for s in scores].index(max_score)
    return max_index, max_score
