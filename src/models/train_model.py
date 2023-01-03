from src.models.bert import *
import src.models as models
import os


def train(model, config, train_arguments, labels_json, test_arguments):
    # If set, run train procedure on a small subset
    if config['train']['use_mini_dataset']:
        train_arguments = train_arguments[:10]
        test_arguments = test_arguments[:10]
    
    print(f"===> Model {model.name}: Training...")
    model.train(train_arguments, labels_json, test_arguments)


def eval(model):
    print(f"===> Model {model.name}: Evaluation...")
    bert_model_evaluation = model.evaluate()
    
    print("-> F1-Scores:")
    for key in bert_model_evaluation['eval_f1-score']:
        print(f"   {key}\t{bert_model_evaluation['eval_f1-score'][key]}")


def get_model(config: dict)->ModelInterface:
    model_name = config['model']['name'].lower()
    if model_name == 'bert':
        return BertModel(config)
    # add new models here
    # elif
    #   ...
    else:
        raise AttributeError(f"found no model class for {config['model']['name']}")