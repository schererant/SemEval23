import pytest
import os
from src.data.make_dataset import load_labels_from_tsv, load_arguments_from_tsv, load_values_from_json, load_json_file

script_dir = os.getcwd()  # <-- absolute dir the script is in
rel_path = "data/raw/"
data_dir = os.path.join(script_dir, rel_path)

filepath_json = os.path.join(data_dir, 'value-categories.json')
filepath_labels = os.path.join(data_dir, 'labels-training.tsv')
filepath_data = os.path.join(data_dir, 'arguments-training.tsv')


def pytest_configure():
    pytest.data_dir = data_dir
    pytest.filepath_json = filepath_json
    pytest.filepath_labels = filepath_labels
    pytest.filepath_data = filepath_data


#%%

# Test hard coded path
def test_path_exists():
    script_dir = os.getcwd()  # <-- absolute dir the script is in
    rel_path = "../data/raw/"
    data_dir = os.path.join(script_dir, rel_path)
    assert os.path.isdir(data_dir)


# TODO: use fixtures
def test_load_labels_from_tsv():
    json_values = load_json_file(filepath_json)
    load_labels_from_tsv(filepath_labels, list(json_values.keys()))
    assert True



