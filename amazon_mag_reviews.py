# -*- coding: utf-8 -*-
"""File to pull Amazon review data for magazine subscriptions and fine tune a BERT model using said data"""

import json
import gzip
from resources.train_model import train_model
from resources.bert_dataset import BertData
from resources.utility_functions import split_data
from transformers import DistilBertTokenizerFast

# set the path to where the data files are stored
PATH_TO_DATA: str = "./data/Magazine_Subscriptions.json.gz"
# features and target columns to pull from the dataset
# we are just demonstrating BERT by trying to classify rating by the review text
# we will leave out other fields to minimize memory usage
FIELDS: list = ['reviewText', 'overall']
# This is the tokenizer we want, we are trying to evaluate sequence rating
# capitalization not as important:
PRETRAINED = 'distilbert_base-uncased'


def read_data(path: str) -> (list, list):
    """This function takes the path to the data and reads it line by line
    :param path - the path str
    :return data, labels - a tuple of 2 lists, one containing text and
    the other containing ratings from all the rows(docs) in the dataset"""
    data = []
    labels = []
    with gzip.open(path) as f:
        for row in f:
            # read row with json library and recieve a dict
            row_dict = json.loads(row)
            # add only relevant data to the data list
            try:
                data.append(row_dict[FIELDS[0]])
                # classes for classificaiton should start from 0 and be in int form
                labels.append(int(row_dict[FIELDS[1]] - 1))
            # ensure that rows with missing reviews are excluded but that the program continues
            except KeyError:
                continue
    # convert list to array for easier/faster processing
    return data, labels


def main():
    """the main method of the script that loads today and trains a fine-tuned BERT model"""
    # collect data matrix and labels
    dataset, labels = read_data(path=PATH_TO_DATA)
    # split both X and y into train, test, and validation sets
    train_data, train_labels, test_data, test_labels, val_data, val_labels = split_data(data=dataset, labels=labels)
    # get tokenizer and create Dataset objects for train, test, and validation sets for use in model training
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    # tokenize the reviews for the training set
    tokens_train = tokenizer(train_data[0:2000], truncation=True, padding=True, max_length=30,
                             add_special_tokens=True)
    # tokenize the validation set
    tokens_val = tokenizer(val_data[0:500], truncation=True, padding=True, max_length=30,
                           add_special_tokens=True)
    review_train = BertData(tokens_train, train_labels[0:2000])
    review_val = BertData(tokens_val, val_labels[0:500])
    train_model(data=review_train, val=review_val, num_labels=len(set(labels)), seq=True)


if __name__ == "__main__":
    main()
