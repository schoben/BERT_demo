"""This file contains helper functions used by both amazon and conll003 scripts"""

import numpy as np
from sklearn.utils import shuffle
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, DistilBertTokenizerFast


def split_data(data: list, labels: list, random_state: int = 42, train_size: int = 0.6) -> (np.array, np.array, np.array):
    """This function creates training, test, and validation sets if needed
    :param data - the array fo data to be shuffled
    :param labels - the labels for the data
    :param random_state - the random state to use if desired, otherwise defaults to 42
    :param train_size - the proportion of data to be used in the training set
    :return train_set, test_set, validation_set - 3 arrays are returned for training, testing, and validiation"""
    # First create training set
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=random_state,
                                                                        test_size=1 - train_size)
    # Then split test set to get validation set
    test_data, test_Labels, validation_data, validation_labels = train_test_split(test_data, test_labels,
                                                                                  random_state=random_state, test_size=0.5)
    return train_data, train_labels, test_data, test_labels, validation_data, validation_labels


def get_tokenizer(name: str = 'distilbert', case: str = 'uncased') -> object:
    """this function takes the name of the relevant tokenizer and returns
    the relevant object for use on the text data
    :param name - the name of the bert model to go with - should usually be distilbert or bert or uncased
    :param case - whether uncased or cased model is used
    :return tokenizer - the tokenizer object to be used"""
    #ensure there is a case if mispelled
    if case != 'cased' or case != 'uncased':
        case = 'uncased'
    if name == 'distilbert':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-' + case)
    elif name == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-' + case)
    else:
        # ensure that an error does not break the program
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-' + case)
    return tokenizer


#def get_max_length(data):



