# -*- coding: utf-8 -*-
"""File to pull Amazon review data for magazine subscriptions and fine tune a BERT model using said data"""

import json
import gzip
from resources.train_model import train_model
from resources.bert_dataset import BertData
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# set the path to where the data files are stored
PATH_TO_DATA: str = "./data/Magazine_Subscriptions.json.gz"
# features and target columns to pull from the dataset
# we are just demonstrating BERT by trying to classify rating by the review text
# we will leave out other fields to minimize memory usage
FIELDS: list = ['reviewText', 'overall']
# This is the tokenizer we want, we are trying to evaluate sequence rating
# capitalization not as important:
PRETRAINED = 'distilbert-base-uncased'


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


def run_seq_cls():
    """the main method of the script that loads today and trains a fine-tuned BERT model"""
    # collect data matrix and labels
    raw_data, raw_labels = read_data(path=PATH_TO_DATA)
    # split both X and y into train, test, and validation sets,ensure data is shuffled
    train_data, val_data, train_label, val_label = train_test_split(raw_data[0:2000],
                                                                    raw_labels[0:2000],
                                                                    shuffle=True,
                                                                    random_state=42)
    # get tokenizer and create Dataset objects for train, test, and validation sets for use in model training
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    # tokenize the reviews for the training set
    tokens_train = tokenizer(train_data[0:2000], truncation=True, padding=True, max_length=30,
                             add_special_tokens=True)
    # tokenize the validation set
    tokens_val = tokenizer(val_data[0:500], truncation=True, padding=True, max_length=30,
                           add_special_tokens=True)
    review_train = BertData(tokens_train, train_label[0:2000])
    review_val = BertData(tokens_val, val_label[0:500])
    train_model(data=review_train, val=review_val, num_labels=len(set(raw_labels)), seq=True)


if __name__ == "__main__":
    run_seq_cls()
