import src.data.make_dataset as md
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

stop_words = list(stopwords.words('english'))

# Define a pipeline combining a text feature extractor with multi lable classifier
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words = stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])


def train_accuracy(labelorder, df_merged):
    for label in list(labelorder.keys()):
        print('... Processing {}'.format(label))
        NB_pipeline.fit(df_merged["Premise"], df_merged[label])
        prediction = NB_pipeline.predict(df_merged["Premise"])
        print('Accuracy is {}'.format(accuracy_score(df_merged[label], prediction)))


label_order = md.load_json_file(filepath_labelorder)
print("Train")
train_accuracy(label_order, df_merged_train)
print("Validation")
train_accuracy(label_order, df_merged_validation)