import argparse
from utils import load_config
import data.make_dataset as make_dataset
import data.format_dataset as format_dataset
import src.models.train_model as train_model
import src.models.predict_model as predict_model
from numpy import arange
import numpy as np
import utils


def setup_parser():

    parser = argparse.ArgumentParser(
        description='SemEval23'
    )
    parser.add_argument('--config', action='store', dest='config', required=True)
    parser.add_argument('--retrain', action='store_true', dest='retrain', default=False)
    # parser.add_argument('--visdom', action='store_true', default=False)

    return parser


# Initialize visdom
# if VISDOM:
#     vis = visdom.Visdom(env=DATASET + ' ' + MODEL)
#     if not vis.check_connection:
#         print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
# else:
#     vis = None


def main():

    ### setup argument parser ###
    parser = setup_parser()
    args = parser.parse_args()


    ### load config ###
    config = load_config(args.config)
    model_name = config["model"]["name"]
    print(f"Run model routine for {model_name} model")


    ### Load data ###
    labels_json = make_dataset.load_json_file('data/raw/value-categories.json')

    df_arguments_training   = make_dataset.load_arguments_from_tsv(config['dataset']['arguments']['train'], default_usage='train')
    df_arguments_validation = make_dataset.load_arguments_from_tsv(config['dataset']['arguments']['val'],   default_usage='validation')
    # df_arguments_test = make_dataset.load_arguments_from_tsv(config['dataset']['arguments']['test'],   default_usage='test')

    df_labels_training = make_dataset.load_labels_from_tsv(config['dataset']['labels']['train'], list(labels_json.keys()))
    df_labels_validation = make_dataset.load_labels_from_tsv(config['dataset']['labels']['val'],   list(labels_json.keys()))
    # df_labels_test = make_dataset.load_labels_from_tsv(config['dataset']['labels']['test'],   list(labels_json.keys()))

    # join arguments and labels
    df_training   = format_dataset.combine_columns(df_arguments_training, df_labels_training)
    df_validation_test = format_dataset.combine_columns(df_arguments_validation, df_labels_validation)
    # df_test = format_dataset.combine_columns(df_arguments_test, df_labels_test)

    # drop usage column
    df_training   = format_dataset.drop_column(df_training, 'Usage')
    df_validation_test = format_dataset.drop_column(df_validation_test, 'Usage')
    # df_test = format_dataset.drop_column(df_test, 'Usage')

    # split validation frame into validation and test frame
    df_validation = df_validation_test.sample(frac=0.67)
    df_test       = df_validation_test.drop(df_validation.index)

    ### Choose model, throws AttributeError if the model name in the config is invalid ###
    model = train_model.get_model(config)
    
    ### train model ###
    # use --retrain flag to run training again, otherwise the last run from the model dir (as specified in config file) is used
    if args.retrain:
        print('Train model...')
        train_model.train(model, config, df_training, list(labels_json.keys()), df_validation)
        train_model.eval(model)
    else:
        print('Skip training model, try to load model from model directory (use --retrain to start training again)')


    print('Run prediction on validation set')
    y_pred, y_true = model.predict(df_validation, config['model']['directory'], list(labels_json.keys()))
    
    print('Optimize threshold on validation data')
    eval_metric = config['evaluate']['metric_th_opt'].split(':')

    thresholds = list(arange(-1.0, 1.0, 0.01))
    scores = [predict_model.scores(y_pred, y_true, tresh=t) for t in thresholds]
    idx_opt, th_opt = utils.get_opt_th(scores, eval_metric)

    # Evaluate on test set
    print('Run prediction on test set')
    y_pred, y_true = model.predict(df_test, config['model']['directory'], list(labels_json.keys()))
    print('Test model and threshold on test set')
    y_pred = np.load('ypredtest.npy')
    y_true = np.load('ytruetest.npy')

    print("Test set results: ", " ".join(eval_metric), predict_model.scores(y_pred, y_true, tresh=thresholds[idx_opt])[eval_metric[0]][eval_metric[1]])

    # Report

    # Visualize

    # Save


if __name__ == '__main__':
    main()
