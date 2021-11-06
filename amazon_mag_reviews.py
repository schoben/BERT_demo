# -*- coding: utf-8 -*-
"""File to pull Amazon review data for magazine subscriptions and fine tune a BERT model using said data"""

import torch
import json
import numpy as np
import gzip
import time

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
    with gzip.open(PATH_TO_DATA) as f:
        for row in f:
            # read row with json library and receieve a dict
            row_dict = json.loads(row)
            # add only relevant data to the data list
            try:
                data.append([row_dict[FIELDS[0]], row_dict[FIELDS[1]]])
            # ensure that rows with missing rewiews are excluded but that the program continues
            except KeyError:
                continue
    return data


if __name__ == "__main__":
    tic = time.perf_counter()
    print(read_data(PATH_TO_DATA)[99])
    toc = time.perf_counter()
    print(toc - tic)
