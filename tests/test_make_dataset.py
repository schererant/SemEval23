import pytest
import os
from src.data.make_dataset import load_labels_from_tsv, load_arguments_from_tsv, load_values_from_json, load_json_file


@pytest.fixture()
def data_dir():
    script_dir = os.getcwd()  # <-- absolute dir the script is in
    rel_path = "../data/raw/"
    return os.path.join(script_dir, rel_path)


@pytest.fixture()
def filepath_json(data_dir):
    return os.path.join(data_dir, 'value-categories.json')


@pytest.fixture()
def filepath_labels(data_dir):
    return os.path.join(data_dir, 'labels-training.tsv')


@pytest.fixture()
def filepath_data(data_dir):
    return os.path.join(data_dir, 'arguments-training.tsv')


def test_path_exists(filepath_json):
    assert os.path.isfile(filepath_json)


def test_load_labels_from_tsv(filepath_json, filepath_labels):
    json_values = load_json_file(filepath_json)
    load_labels_from_tsv(filepath_labels, list(json_values.keys()))
    assert True


def test_load_arguments_from_tsv(filepath_data):
    load_arguments_from_tsv(filepath_data)
    assert True



