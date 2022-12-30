import sys
from src.data.make_dataset import load_values_from_json, load_arguments_from_tsv, load_json_file, load_labels_from_tsv#, combine_columns, split_arguments
from src.models.bert import *
from src.data.format_dataset import combine_columns, split_arguments

VALIDATE = False


def main():
    print("Hello world")

    # # format dataset
    # df_train_all = []
    # df_valid_all = []

    # load values
    json_values = load_json_file('data/raw/value-categories.json')

    # load arguments
    df_arguments = load_arguments_from_tsv('data/processed/arguments-training_unify_stance.tsv', default_usage='train')

    # load labels
    df_labels = load_labels_from_tsv('data/raw/labels-training.tsv', list(json_values.keys()))
    # join arguments and labels
    df_full_level = combine_columns(df_arguments, df_labels)
    # split dataframe by usage
    train_arguments, valid_arguments, _ = split_arguments(df_full_level)
    # df_train_all.append(train_arguments)
    # df_valid_all.append(valid_arguments)

    print("===> Bert: Training...")
    if VALIDATE:
        # bert_model_evaluation = train_bert_model(df_train_all,
        #                                          os.path.join(model_dir, 'bert_train_level{}'.format(levels[i])),
        #                                          values[levels[i]], test_dataframe=df_valid_all[i])
        print("F1-Scores:")
        # print(bert_model_evaluation['eval_f1-score'])
    else:
        train_bert_model(train_arguments, 'models/bert_train', list(json_values.keys()))


if __name__ == '__main__':
    main()
