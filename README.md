# SemEval23
SemEval 2023 Task 4. ValueEval: Identification of Human Values behind Arguments

# Report  
For the report, an Overleaf project has been created with the provided template:  
https://www.overleaf.com/6797958162jgjxzkwrsfkr  

# Get started
### Download the data

### Setup the environment

### Run the code

# Structure
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Make this project pip installable with `pip install -e`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

# Project desription
## Task
Given a textual argument and a human value category, classify whether or not the argument draws on that category. This task uses a set of 20 value categories compiled from the social science literature and described in our ACL paper. Participants can submit runs (also known as approaches or systems) that detect one, a subset, or all of these values in arguments. Arguments are given as premise text, conclusion text, and binary stance of the premise to the conclusion ("in favor of" or "against").

If you want to focus on a few categories but are unsure which to pick, we suggest to take (from) the most frequent ones in our initial dataset: Self-direction: action, Achievement, Security: personal, Security: societal, Benevolence: caring, Universalism: concern.

## Data
Data is provided as tab-separated values files with one header line. Stay up-to-date and report problems on the task mailing list.

The arguments-training/validation/test.tsv files contain one argument per line: its unique argument ID, the conclusion, the premise's stance towards the conclusion, and the premise itself. Example with tab-separated columns highlighted:

`
> Argument ID	Conclusion	Stance	Premise
> A01010	We should prohibit school prayer	against	it should be allowed if the student wants to pray as long as it is not interfering with his classes
> A01011	We should abolish the three-strikes laws	in favor of	three strike laws can cause young people to be put away for life without a chance to straight out their life
> A01012	The use of public defenders should be mandatory	in favor of	the use of public defenders should be mandatory because some people don't have money for a lawyer and this would help those that don't
`

The labels-training/validation/test.tsv files also contain one argument per line: its unique argument ID and one column for each of the 20 value categories with a 1 meaning that the argument resorts to the value category and a 0 that not. Example with tab-separated columns highlighted:

`
Argument ID	Self-direction: thought	Self-direction: action	Stimulation	Hedonism	Achievement	Power: dominance	Power: resources	Face	Security: personal	Security: societal	Tradition	Conformity: rules	Conformity: interpersonal	Humility	Benevolence: caring	Benevolence: dependability	Universalism: concern	Universalism: nature	Universalism: tolerance	Universalism: objectivity
A01010	1	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	1	0	0	0
A01011	0	0	0	0	1	0	0	1	0	0	0	0	0	0	1	0	0	0	1	1
A01012	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
`

## Evaluation

Runs are evaluated on the basis of F1-score, Precision, and Recall in TIRA: averaged over all value categories and for each category individually. For the shared task leaderboard, the runs are ranked according to the averaged F1-score on the "arguments-test" dataset, corresponding to the arguments-test.tsv of the dataset.

## Submission

This task uses TIRA for submissions, which allows for both run file upload and Docker image submission. The latter allows for excellent reproducibility of your run. Registered teams will receive links to register with TIRA up to one work day after registration.

Each registered team can submit up to 4 different runs on the "arguments-test" dataset. Teams are allowed to submit one additional "early bird" run to this dataset until December 16. If you submit more runs, please tell us which one you would like to have scored. You can submit an arbitrary amount of runs to the other datasets. The submission format is the same as the labels-training/validation/test.tsv with value category columns being optional (skip categories for which the run contains no prediction).

To submit a run file, go to this task's TIRA page. Log in. Click "SUBMIT". Make sure "UPLOAD SUBMISSION" is selected. Specify your file and the dataset and "UPLOAD".

To submit a Docker image, go to this task's TIRA page. Log in. Click "SUBMIT". Make sure "DOCKER SUBMISSION" is selected. Click on "UPLOAD IMAGES" and follow the instructions. Once uploaded, click on "ADD CONTAINER". Specify the command and the Docker image and "ADD CONTAINER". A randomly named tab is created. Click on it, specify the dataset and "RUN CONTAINER". You can copy the random baseline to get started with Docker submissions.

For a quick comparison, we already run the random baseline and 1-baseline on the training and validation sets using the team name "aristotle".

In case of problems or questions concerning TIRA, please use the TIRA forum. Note: At the moment, the TIRA web interface sometimes fails to provide feedback on actions. Reload the page in such cases.
