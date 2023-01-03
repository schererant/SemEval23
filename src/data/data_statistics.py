import pandas as pd
import make_dataset as md
from pathlib import Path
import os
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from statistics import mean, stdev, median


def stance_distribution(dataframe):
    vals = dataframe['Stance'].value_counts()
    n_rows = dataframe.shape[0]
    print(f"   Total values: {n_rows}")
    for key in vals.keys():
        print(f"   {key}: {vals[key]} ({vals[key]/n_rows*100:.1f}%)")


def word_counts(dataframe, set, saveto=None):
    words = lambda s: len(s.split())

    fig, (ax1, ax2) = plt.subplots(2, 1)

    fig.set_figheight(8)
    fig.set_figwidth(7)
    
    t = dataframe['Conclusion'].map(words)
    
    ax1.text(20, 3500, f"Median: {median(t)} words\nper conclusion", bbox=dict(facecolor='none', edgecolor='red'))
    
    ax1.hist(t, bins=10)
    ax1.set_title(f'Word counts: conclusion {set} set')
    ax1.set_ylabel('occurances')
    
    t = dataframe['Premise'].map(words)
    ax2.text(68, 2120, f"Median: {median(t)} words\nper premise", bbox=dict(facecolor='none', edgecolor='red'))
    ax2.hist(t, bins=10)
    ax2.set_title(f'Word counts: premise {set} set')
    ax2.set_xlabel('number of words')
    ax2.set_ylabel('occurances')
    
    plt.show()

    if not saveto is None:
        fig.savefig(saveto)


def word_cloud(dataframe, column, set, saveto=None):
    words = []

    for stance in ['in favor of', 'against']:
        for row in dataframe[dataframe['Stance']==stance][column]:
            words.append(row)

        words_string = (" ").join(words)
        words_list = words_string.split()

        n = 10
        print(f"{n} Most frequent words in {set} data ({stance}):")
        word_frame = pd.DataFrame({"words": words_list})
        print(word_frame.value_counts().index.tolist()[:n])

        print(f"Plot {set} data wordcloud for {len(words_list)} words")

        cloud = WordCloud().generate(words_string)
        
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis("off")
        if not saveto is None:
            path = str(saveto).rsplit('.', 1)
            path.insert(-1, '_'+stance+'.')
            plt.savefig(''.join(path))
        plt.show()


def label_distribution(dataframe, saveto=None):
    sum_label     = dataframe[dataframe.columns[1:]].sum(axis=0)
    sum_dp = dataframe[dataframe.columns[1:]].sum(axis=1)

    dp_mean = mean(sum_dp)
    dp_std = stdev(sum_dp)

    indexes = list(map(lambda s: s.split(':')[-1], sum_label.index))

    _, ax1 = plt.subplots()

    ax1.bar(range(len(indexes)), sum_label)
    ax1.set_title('Distribution of train labels')
    ax1.set_ylabel('Occurances')
    ax1.text(0.5, 1800, f"{dp_mean:.2f}$\pm${dp_std:.2f} labels\nper datapoint", bbox=dict(facecolor='none', edgecolor='red'))
    plt.xticks(range(len(indexes)), indexes, rotation=90)
    plt.tight_layout()
    if not saveto is None:
        plt.savefig(saveto)
    plt.show()


if __name__ == '__main__':
    data_dir_processed = os.path.dirname(__file__) / Path('../../data/processed')
    data_dir_raw = os.path.dirname(__file__) / Path('../../data/raw')
    report_dir = os.path.dirname(__file__) / Path('../../reports/statistical_analysis')
    
    train_frame = md.load_arguments_from_tsv(data_dir_processed/'arguments-training_unify_stance.tsv', 'train')

    value_categories = md.load_json_file(data_dir_raw/'value-categories.json')
    train_labels = md.load_labels_from_tsv(data_dir_raw/'labels-training.tsv', list(value_categories.keys()))

    #label_distribution(train_labels, saveto=report_dir/'label_distribution_train.png')
    #stance_distribution(train_frame)

    word_counts(train_frame, 'train', saveto=report_dir/'wordlength_conclusion_premise_train.png')

    #word_cloud(train_frame, 'Premise', 'train', saveto=report_dir/'wordcloud_premise_train.png')
