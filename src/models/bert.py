import torch
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, DistilBertModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import (Dataset, DatasetDict, load_dataset)
import utils
import models.predict_model as predict_model
from models.model_interface import ModelInterface

"""
The folllwoing code is taken from https://github.com/webis-de/acl22-identifying-the-human-values-behind-arguments/blob/main/src/python/components/models/bert.py

"""
class MultiLabelTrainer(Trainer):
    """
        A transformers `Trainer` with custom loss computation
        Methods
        -------
        compute_loss(model, inputs, return_outputs=False):
            Overrides loss computation from Trainer class
        """
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


class BertModel(ModelInterface):
    def __init__(self, config) -> None:
        self.name = config['model']['name']

        self.model_dir = config['model']['directory']

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.train_args = TrainingArguments(
                output_dir=self.model_dir,
                evaluation_strategy=config['evaluate']['strategy'],
                eval_steps=config['evaluate']['eval_steps'],
                learning_rate=config['optimizer']['lr'],
                per_device_train_batch_size=config['train']['batch_size'],
                per_device_eval_batch_size=config['train']['batch_size'],
                num_train_epochs=config['train']['epochs'],
                weight_decay=config['train']['weight_decay'],
                load_best_model_at_end=config['evaluate']['reload_best_model_weights'],
                metric_for_best_model=config['evaluate']['metric'],
                save_strategy=config['train']['save_strategy'],
                save_steps=config['train']['save_steps'],
                seed=config['train']['seed'],
                lr_scheduler_type=config['optimizer']['scheduler_type']
            )

        self.predict_args = TrainingArguments(
            output_dir=self.model_dir,
            do_train=False,
            do_eval=False,
            do_predict=True,
            per_device_eval_batch_size=config['evaluate']['batch_size']
        )

    def train(self, train_dataframe, labels, test_dataframe=None):
        """
        Trains Bert model with the arguments in `train_dataframe`
        Parameters
        ----------
        train_dataframe: pd.DataFrame
            The arguments to be trained on
        model_dir: str
            The directory for storing the trained model
        labels : list[str]
            The labels in the training data
        test_dataframe: pd.DataFrame, optional
            The validation arguments (default is None)
        num_train_epochs: int, optional
            The number of training epochs (default is 20)
        Returns
        -------
        Metrics
            result of validation if `test_dataframe` is not None
        NoneType
            otherwise
        """
        if test_dataframe is None:
            test_dataframe = train_dataframe
        
        ds, labels = self.convert_to_dataset(train_dataframe, test_dataframe, labels)

        model = utils.load_model_from_data_dir("bert-base-uncased", num_labels=len(labels))

        self.multi_trainer = MultiLabelTrainer(
            model,
            self.train_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            compute_metrics=lambda x: predict_model.compute_metrics(x, labels),
            tokenizer=self.tokenizer
        )

        self.multi_trainer.train()

        model.save_pretrained(self.model_dir)

    def evaluate(self):
        print("Evaluating model")
        return self.multi_trainer.evaluate()

    def convert_to_dataset(self, train_dataframe, test_dataframe, labels):
        """
            Converts pandas DataFrames into a DatasetDict
            Parameters
            ----------
            train_dataframe : pd.DataFrame
                Arguments to be listed as "train"
            test_dataframe : pd.DataFrame
                Arguments to be listed as "test"
            labels : list[str]
                The labels in both DataFrames
            Returns
            -------
            tuple(DatasetDict, list[str])
                a `DatasetDict` with attributes "train" and "test" for the listed arguments,
                a `list` with the contained labels
            """
        column_intersect = [x for x in (['Premise'] + labels) if x in train_dataframe.columns.values]

        train_dataset = Dataset.from_dict((train_dataframe[column_intersect]).to_dict('list'))
        test_dataset = Dataset.from_dict((test_dataframe[column_intersect]).to_dict('list'))

        ds = DatasetDict()
        ds['train'] = train_dataset
        ds['test'] = test_dataset

        ds = ds.map(lambda x: {"labels": [int(x[c]) for c in ds['train'].column_names if
                                        c not in ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Part']]})

        cols = ds['train'].column_names
        cols.remove('labels')

        ds_enc = ds.map(self.tokenize_and_encode, batched=True, remove_columns=cols)

        cols.remove('Premise')

        return ds_enc, cols


    def predict(self, dataframe, model_dir, labels):
        """
        Classifies each argument using the Bert model stored in `model_dir`
        Parameters
        ----------
        dataframe: pd.Dataframe
            The arguments to be classified
        model_dir: str
            The directory of the pre-trained Bert model to use
        labels: list[str]
            The labels to predict
        Returns
        -------
        np.ndarray
            numpy nd-array with the predictions given by the model
        """
        ds, no_labels = self.convert_to_dataset(dataframe, dataframe, labels)
        num_labels = len(labels)
        ds = ds.remove_columns(['labels'])

        model = utils.load_model_from_data_dir(model_dir, num_labels=num_labels)

        multi_trainer = MultiLabelTrainer(
            model,
            self.predict_args,
            tokenizer=self.tokenizer
        )

        prediction = 1 * (multi_trainer.predict(ds['train']).predictions > 0.5)

        return prediction

    def tokenize_and_encode(self, examples):
        """Tokenizes each arguments "Premise" """
        return self.tokenizer(examples['Premise'], truncation=True)


    def convert_to_dataset(self, train_dataframe, test_dataframe, labels):
        """
            Converts pandas DataFrames into a DatasetDict
            Parameters
            ----------
            train_dataframe : pd.DataFrame
                Arguments to be listed as "train"
            test_dataframe : pd.DataFrame
                Arguments to be listed as "test"
            labels : list[str]
                The labels in both DataFrames
            Returns
            -------
            tuple(DatasetDict, list[str])
                a `DatasetDict` with attributes "train" and "test" for the listed arguments,
                a `list` with the contained labels
            """
        column_intersect = [x for x in (['Premise'] + labels) if x in train_dataframe.columns.values]

        train_dataset = Dataset.from_dict((train_dataframe[column_intersect]).to_dict('list'))
        test_dataset = Dataset.from_dict((test_dataframe[column_intersect]).to_dict('list'))

        ds = DatasetDict()
        ds['train'] = train_dataset
        ds['test'] = test_dataset

        ds = ds.map(lambda x: {"labels": [int(x[c]) for c in ds['train'].column_names if
                                        c not in ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Part']]})

        cols = ds['train'].column_names

        cols.remove('labels')

        ds_enc = ds.map(self.tokenize_and_encode, batched=True, remove_columns=cols)
        
        cols.remove('Premise')
        
        return ds_enc, cols


