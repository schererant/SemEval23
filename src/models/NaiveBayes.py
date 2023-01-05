import src.data.make_dataset as md
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from model_interface import ModelInterface

class NBModel(ModelInterface):
    def __init__(self, config):
        self.config = config

    def init_pipeline(self):
        """ Initialization pipeline with removing stop words and having a one vs rest classifier.
        """
        stop_words = list(stopwords.words('english'))
        return Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])

    def train(self, train_frame, label_columns):
        """Defining a pipeline combining a text feature extractor with multi label classifier
        """
        self.label_columns = label_columns
        self.NB_pipelines = {label_col: self.init_pipeline() for label_col in label_columns}

        for label_col in label_columns:
            self.NB_pipelines[label_col].fit(train_frame['Premise'], train_frame[label_col])

    def predict(self, dataframe):
        """Predicting the labels"""
        preds = {}

        for col in self.label_columns:
            preds[col] = self.NB_pipelines[col].predict(dataframe['Premise'])

        pred_df = pd.DataFrame(preds)

        return pred_df

    def evaluate(self, data_frame):
        """"Evaluating the labels, a small evaluation. A more elaborate evaluation can be found in the evaluation code. """
        pred_df = self.predict(data_frame)
        scores = {col: accuracy_score(pred_df[col], data_frame[col]) for col in self.label_columns}

        print(f"Accuracy score: {scores}")
        return scores


if __name__ == "__main__":
    def df_dataset(filepath_arguments, filepath_labelorder, filepath_labels):
        df_arguments = md.load_arguments_from_tsv(filepath_arguments)
        label_order = md.load_json_file(filepath_labelorder)
        df_labels = md.load_labels_from_tsv(filepath_labels, list(label_order.keys()))
        df_merged = md.merge_arg_label(df_arguments, df_labels)
        return df_merged


    filepath_arguments_train = "../../data/processed/arguments-training_unify_stance.tsv"
    filepath_labelorder = "../../data/raw/value-categories.json"
    filepath_labels_train = "../../data/raw/labels-training.tsv"

    filepath_arguments_val = "../../data/processed/arguments-validation_unify_stance.tsv"
    filepath_labels_val = "../../data/raw/labels-validation.tsv"

    df_merged_train = df_dataset(filepath_arguments_train, filepath_labelorder, filepath_labels_train)
    df_merged_validation = df_dataset(filepath_arguments_val, filepath_labelorder, filepath_labels_val)

    label_order = md.load_json_file(filepath_labelorder)
    label_columns = label_order.keys()

    nb_model = NBModel()
    nb_model.train(df_merged_train, label_columns)
    print(nb_model.predict(df_merged_validation))
    nb_model.evaluate(df_merged_validation)