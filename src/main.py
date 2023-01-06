import argparse
from utils import load_config
import data.make_dataset as make_dataset
import data.format_dataset as format_dataset
import src.models.train_model as train_model
import src.models.predict_model as predict_model
import os
import json


def setup_parser():
    # Argument parser for command line options
    parser = argparse.ArgumentParser(
        description='SemEval23'
    )
    parser.add_argument('--config', action='store', dest='config', required=True)
    parser.add_argument('--retrain', action='store_true', dest='retrain', default=False)

    return parser


def main():

    ### setup argument parser ###
    parser = setup_parser()
    args = parser.parse_args()


    ### load config (set by command line) ###
    config = load_config(args.config)
    model_name = config["model"]["name"]
    print(f"Run model routine for {model_name} model")


    ### Load data (train and test data) ###
    labels_json = make_dataset.load_json_file('data/raw/value-categories.json')

    df_arguments_training   = make_dataset.load_arguments_from_tsv(config['dataset']['arguments']['train'], default_usage='train')
    df_arguments_test = make_dataset.load_arguments_from_tsv(config['dataset']['arguments']['val'],   default_usage='validation')

    df_labels_training = make_dataset.load_labels_from_tsv(config['dataset']['labels']['train'], list(labels_json.keys()))
    df_labels_test = make_dataset.load_labels_from_tsv(config['dataset']['labels']['val'],   list(labels_json.keys()))

    # join arguments and labels
    df_training_val   = format_dataset.combine_columns(df_arguments_training, df_labels_training)
    df_test = format_dataset.combine_columns(df_arguments_test, df_labels_test)

    # drop usage column
    df_training_val   = format_dataset.drop_column(df_training_val, 'Usage')
    df_test = format_dataset.drop_column(df_test, 'Usage')

    if config['evaluate']['split_validation_set']:
        # split validation frame into validation and test frame
        df_training   = df_training_val.sample(frac=0.7, random_state=config['train']['seed'])
        df_validation = df_training_val.drop(df_training.index)
    else:
        # do not use a separate validation set for evaluation
        df_training = df_training_val
        df_validation = None

    if config['train']['use_mini_dataset']:
        # if specified in config, use only a small data set
        # to speed up the process for testing purpose
        df_training   = df_training[:10]
        df_test       = df_test[:10]
        if not df_validation is None:
            df_validation = df_validation[:10]

    ### Choose model, throws AttributeError if the model name in the config file is invalid ###
    model = train_model.get_model(config)
    
    ### train model ###
    # use --retrain flag to run training again
    # otherwise the last trained model from the model-directory (as specified in config file) is used
    # due to fast training time, the NaiveBayes model is not saved, and thus is always retrained
    if args.retrain or config['train']['always_retrain']:
        print('Train model...')
        model.train(df_training, list(labels_json.keys()), df_validation)
    else:
        print('Skip training model (use --retrain to start model training again)')

    # get evaluation cirterion from config file
    eval_metric = config['evaluate']['metric_th_opt'].split(':')

    # run prediction on training data
    y_pred_train, y_true_train = model.predict(df_training, config['model']['directory'], list(labels_json.keys()))

    if not df_validation is None:
        # if there is a separete validation set, predict on it and use it for parameter optimization (threshold for bert, nothing for NB)
        print('Run prediction on validation set')
        y_pred_val, y_true_val = model.predict(df_validation, config['model']['directory'], list(labels_json.keys()), use_threshold=False)

        # run optimization on validation set, e.g. threshold optimization
        model.optimize(y_pred_val, y_true_val, eval_metric)

    # Evaluate the model after optimization on test set
    print('Run prediction on test set')
    y_pred_test, y_true_test = model.predict(df_test, config['model']['directory'], list(labels_json.keys()))
    
    # Compute the scopres and save the to a file for test and training data
    scores_train = predict_model.scores(y_pred_train, y_true_train)
    print("Train set results: ", " ".join(eval_metric), scores_train[eval_metric[0]][eval_metric[1]])
    
    scores_test = predict_model.scores(y_pred_test, y_true_test)
    print("Test set results: ",  " ".join(eval_metric), scores_test[eval_metric[0]][eval_metric[1]])

    with open(os.path.join(config['model']['directory'], f"model_{config['model']['name']}_scores.json"), 'w') as f:
        json.dump({'train': scores_train, 'test': scores_test}, f, indent=4)


if __name__ == '__main__':
    main()
