import pytest
import os
from src.data.make_dataset import load_labels_from_tsv, load_arguments_from_tsv, load_values_from_json, load_json_file


@pytest.fixture
def filepath_json():
    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir, '/data/raw/')
    filepath = str(data_dir + 'value-categories.json')
    return filepath


@pytest.fixture
def filepath_labels():
    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir, '/data/raw/')
    return os.path.join(data_dir, 'labels-training.tsv')


@pytest.fixture
def filepath_data():
    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir, '/data/raw/')
    return os.path.join(data_dir, 'arguments-training.tsv')


# Test hard coded path
def test_path_exists():
    assert os.path.isdir('/data/')


# TODO: use fixtures
def test_load_labels_from_tsv():
    filepath_json = '/data/raw/value-categories.json'
    filepath_labels = '/data/raw/labels-training.tsv'
    # curr_dir = os.getcwd()
    # data_dir = os.path.join(curr_dir, '/data/raw/')
    # # filepath_json = os.path.join(data_dir, 'value-categories.json')
    # filepath_json = data_dir + 'value-categories.json'
    json_values = load_json_file(filepath_json)
    load_labels_from_tsv(filepath_labels, list(json_values.keys()))
    assert True



