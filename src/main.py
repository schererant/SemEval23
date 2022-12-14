import argparse
from utils import load_config
import os

# Specify config name
CONFIG_NAME = 'baseline_config.yaml'

# Load config file
config = load_config(CONFIG_NAME)

# Define main argument for parser
parser = argparse.ArgumentParser(
    description='SemEval23'
)

parser.add_argument('--visdom', action='store_true', default=False)

args = parser.parse_args()

# Initialize visdom
# if VISDOM:
#     vis = visdom.Visdom(env=DATASET + ' ' + MODEL)
#     if not vis.check_connection:
#         print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
# else:
#     vis = None


def main():
    print("Hello world")

    # Set variables (config and parser

    # Load data

    # Choose model

    # Train

    # Validate

    # Report

    # Visualize

    # Save


if __name__ == '__main__':
    main()
