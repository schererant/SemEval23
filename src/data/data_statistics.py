import pandas as pd
import make_dataset as md
from pathlib import Path
import os
from matplotlib import pyplot as plt
from wordcloud import WordCloud


def label_distribution(dataframe):
    vals = dataframe['Stance'].value_counts()
    n_rows = dataframe.shape[0]
    print(f"   Total values: {n_rows}")
    for key in vals.keys():
        print(f"   {key}: {vals[key]} ({vals[key]/n_rows*100:.1f}%)")


def word_counts(dataframe, set, saveto=None):
    words = lambda s: len(s.split())

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_figheight(6)
    fig.set_figwidth(9)

    t = dataframe['Conclusion'].map(words)
    ax1.hist(t, bins=10)
    ax1.set_title(set + ' conclusion word counts')
    ax1.set_xlabel('number of words')
    ax1.set_ylabel('occurances')
    
    t = dataframe['Premise'].map(words)
    ax2.hist(t, bins=10)
    ax2.set_title(set + ' premise word counts')
    ax2.set_xlabel('number of words')
    
    plt.show()

    if not saveto is None:
        fig.savefig(saveto)


def word_cloud(dataframe, set, saveto=None):
    words = []

    for row in dataframe['Conclusion']:
        words.append(row)
    
    words = (" ").join(words)
    print(f"Plot {set} wordcloud for {len(words)} words")

    cloud = WordCloud().generate(words)
    
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    if not saveto is None:
        plt.savefig(saveto)
    plt.show()


if __name__ == '__main__':
    data_dir = os.path.dirname(__file__) / Path('../../data/processed')
    report_dir = os.path.dirname(__file__) / Path('../../reports')
    
    train_frame = md.load_arguments_from_tsv(data_dir/'arguments-training_unify_stance.tsv', 'train')
    val_frame   = md.load_arguments_from_tsv(data_dir/'arguments-validation_unify_stance.tsv', 'validation')
    
    print("Print the label distribution...")
    print("Train frame")
    label_distribution(train_frame)
    print("Val frame")
    label_distribution(val_frame)

    print("Plot histogram of word lengths...")
    word_counts(train_frame, 'train', saveto=report_dir/'wordlength_hist_trainframe.png')
    word_counts(val_frame, 'validation', saveto=report_dir/'wordlength_hist_valframe.png')

    print("Plot word clouds...")
    word_cloud(train_frame, 'train', saveto=report_dir/'wordcloud_trainframe.png')
    word_cloud(val_frame, 'validation', saveto=report_dir/'wordcloud_valframe.png')
