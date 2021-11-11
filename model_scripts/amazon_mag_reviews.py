# -*- coding: utf-8 -*-
"""File to pull Amazon review data for magazine subscriptions and fine tune a BERT model using said data"""


import json
import gzip
from resources.train_model import train_model
from resources.bert_dataset import BertData
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# features and target columns to pull from the dataset
# we are just demonstrating BERT by trying to classify rating by the review text
# we will leave out other fields to minimize memory usage
FIELDS: list = ['reviewText', 'overall']
# This is the tokenizer we want, we are trying to evaluate sequence rating
# capitalization not as important:
PRETRAINED = 'distilbert-base-uncased'


def read_data(path: str, trunc: int = None) -> (list, list):
    """This function takes the path to the data and reads it line by line
    :param path - the path str
    :param trunc - ann optional parameter, limits the number of samples to be collected.
    :return data, labels - a tuple of 2 lists, one containing text and
    the other containing ratings from all the rows(docs) in the dataset"""
    data = []
    labels = []
    with gzip.open(path) as f:
        for row in f:
            # read row with json library and receive a dict
            row_dict = json.loads(row)
            # add only relevant data to the data list
            try:
                data.append(row_dict[FIELDS[0]])
                # classes for classification should start from 0 and be in int form
                labels.append(int(row_dict[FIELDS[1]] - 1))
            # ensure that rows with missing reviews are excluded but that the program continues
            except KeyError:
                continue
            if trunc:
                if len(data) == trunc:
                    break
    return data, labels


def run_seq_cls(path: str, trunc: int = None):
    """the main method of the script that loads today and trains a fine-tuned BERT model
    :param path - a string where the file location is.
    :param trunc - an optional parameter, limits the number of samples to be collected."""
    # collect data matrix and labels
    raw_data, raw_labels = read_data(path=path, trunc=trunc)
    # split both X and y into train, test, and validation sets,ensure data is shuffled
    train_data, val_data, train_label, val_label = train_test_split(raw_data,
                                                                    raw_labels,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    test_size=0.25)
    # get tokenizer and create Dataset objects for train, test, and validation sets for use in model training
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    # tokenize the reviews for the training set
    tokens_train = tokenizer(train_data, truncation=True, padding=True, max_length=30,
                             add_special_tokens=True)
    # tokenize the validation set
    tokens_val = tokenizer(val_data, truncation=True, padding=True, max_length=30,
                           add_special_tokens=True)
    # The resulting features training tensor will be 2 dimensional and contain
    # sequences (rows) with 30 tokens encoded numerically each as specified by the max_length
    # Sequences shorter than 30 tokens will be padded with zeros.
    # There is one label per sequence which represents a star rating for that document.
    # This is in contrast to the NER data, which has a label for each token and so the target tensor
    # is 2d.
    review_train = BertData(tokens_train, train_label)
    review_val = BertData(tokens_val, val_label)
    train_model(data=review_train, val=review_val, num_labels=len(set(raw_labels)), seq=True)
