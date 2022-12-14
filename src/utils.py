import yaml
import os


# Load yaml config file
def load_config(config_name):

    with open(os.path.join('../config/', config_name)) as file:
        config = yaml.safe_load(file)

    return config

