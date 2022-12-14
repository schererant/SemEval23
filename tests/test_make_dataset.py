import pytest
import os
from src.data.make_dataset import load_labels_from_tsv, load_arguments_from_tsv, load_values_from_json, load_json_file
from pathlib import Path

m_smaples = 5393

@pytest.fixture()
def raw_data_dir() -> Path:
    return Path("data/raw/")


@pytest.fixture()
def filepath_value_categories_json(raw_data_dir):
    return raw_data_dir / 'value-categories.json'


@pytest.fixture()
def filepath_labels_training_tsv(raw_data_dir):
    return raw_data_dir / 'labels-training.tsv'


@pytest.fixture()
def filepath_data(raw_data_dir):
    return raw_data_dir / 'arguments-training.tsv'


def check_dataframe_cols(df, columns):
    for c in df.columns.values:
        try:
            columns.pop(columns.index(c))
        except ValueError:
            print("got collumn in dataframe that is not present in the given list")
            return False
    
    return len(columns) == 0


def test_categories_json_exists(filepath_value_categories_json):
    assert os.path.isfile(filepath_value_categories_json)


def test_training_labels_exist(filepath_labels_training_tsv):
    assert os.path.isfile(filepath_labels_training_tsv)


def test_load_labels_from_tsv(filepath_value_categories_json, filepath_labels_training_tsv):
    json_values = load_json_file(filepath_value_categories_json)

    assert(len(json_values) == 20)

    dataframe = load_labels_from_tsv(filepath_labels_training_tsv, list(json_values.keys()))

    expected_columns = [
        'Argument ID', 'Self-direction: thought', 'Self-direction: action', 
        'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance', 
        'Power: resources', 'Face', 'Security: personal', 'Security: societal',
        'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 
        'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern',
        'Universalism: nature', 'Universalism: tolerance', 'Universalism: objectivity',
    ]

    assert dataframe.shape == (m_smaples, len(expected_columns))
    assert check_dataframe_cols(dataframe, expected_columns)



def test_load_arguments_from_tsv(filepath_data):
    dataframe = load_arguments_from_tsv(filepath_data)

    expected_columns = ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Usage']

    assert dataframe.shape == (m_smaples, len(expected_columns))
    assert check_dataframe_cols(dataframe, expected_columns)
