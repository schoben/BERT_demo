# -*- coding: utf-8 -*-
"""File to pull Amazon review data for magazine subscriptions and fine tune a BERT model using said data"""

import torch
import torch.nn as nn
import json
import numpy as np
import gzip
import time
from resources.train_model import train_model
from resources.bert_dataset import BertData
from resources.utility_functions import split_data, get_tokenizer



# set the path to where the data files are stored
PATH_TO_DATA: str = "./data/Magazine_Subscriptions.json.gz"
# features and target columns to pull from the dataset
# we are just demonstrating BERT by trying to classify rating by the review text
# we will leave out other fields to minimize memory usage
FIELDS: list = ['reviewText', 'overall']


def read_data(path: str) -> list:
    """This function takes the path to the data and reads it line by line
    :param path - the path str
    :return data - a list containing all the rows from the dataset"""
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
    #split both X and y into train, test, and validation sets
    train_data, train_labels,  test_data, test_labels, val_data, val_labels = split_data(data=dataset, labels=labels)
    # get tokenizer and create Dataset objects for train, test, and validation sets for use in model training
    tokenizer = get_tokenizer()
    tokens_train = tokenizer(train_data[0:2000], truncation=True, padding=True, max_length=30, add_special_tokens=True)
    review_train = BertData(tokens_train, train_labels[0:2000])
    train_model(data=review_train, num_labels=len(set(labels)), seq=True)






if __name__ == "__main__":
    main()
