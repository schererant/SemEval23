from pathlib import Path
from pandas import read_csv
import os


def change_spelling(src_dir, file_names, tgt_dir):
    """
    Change spelling of lables in the raw data sets to unique spelling
    -> change spelling of 'in favour of' to 'in favor of'
    """

    src_dir = os.path.dirname(__file__)/Path(src_dir)
    tgt_dir = os.path.dirname(__file__)/Path(tgt_dir)

    for file_name in file_names:
        try:
            df = read_csv(src_dir/file_name, encoding='utf-8', sep='\t', header=0)

            df['Stance'] = df['Stance'].replace('in favour of', 'in favor of')

            assert 'in favour of' not in list(df['Stance'].unique())

            new_file_name = file_name.split('.')[0] + '_unify_stance.tsv'

            df.to_csv(tgt_dir/new_file_name, sep='\t', encoding='utf-8')

        except FileNotFoundError as e:
            print(e.strerror, src_dir/file_name)
            continue


if __name__ == '__main__':
    change_spelling('../../data/raw', ('arguments-test.tsv', 'arguments-training.tsv', 'arguments-validation.tsv'), '../../data/processed')