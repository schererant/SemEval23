import argparse
from utils import load_config
import data.make_dataset as make_dataset
import data.format_dataset as format_dataset
import src.models.train_model as train_model
import sys

def setup_parser():

    parser = argparse.ArgumentParser(
        description='SemEval23'
    )
    parser.add_argument('--config', action='store', dest='config', required=True)
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
    df_arguments_test = make_dataset.load_arguments_from_tsv(config['dataset']['arguments']['test'],   default_usage='test')

    df_labels_training = make_dataset.load_labels_from_tsv(config['dataset']['labels']['train'], list(labels_json.keys()))
    df_labels_validation = make_dataset.load_labels_from_tsv(config['dataset']['labels']['val'],   list(labels_json.keys()))
    # df_labels_test = make_dataset.load_labels_from_tsv(config['dataset']['labels']['test'],   list(labels_json.keys()))

    # join arguments and labels
    df_training   = format_dataset.combine_columns(df_arguments_training, df_labels_training)
    df_validation = format_dataset.combine_columns(df_arguments_validation, df_labels_validation)
    # df_test = format_dataset.combine_columns(df_arguments_test, df_labels_test)

    # drop usage column
    df_training   = format_dataset.drop_column(df_training, 'Usage')
    df_validation = format_dataset.drop_column(df_validation, 'Usage')
    # df_test = format_dataset.drop_column(df_test, 'Usage')

    ### Choose model, throws AttributeError if the model name in the config is invalid ###
    model = train_model.get_model(config)
    
    ### train model ###
    train_model.train(model, config, df_training, list(labels_json.keys()), df_validation)

    # Validate
    train_model.eval(model)

    # Report

    # Visualize

    # Save


if __name__ == '__main__':
    main()
