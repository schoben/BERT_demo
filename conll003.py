# -*- coding: utf-8 -*-
"""This file loads data from the conll003 dataset and uses a fine-tuned BERT model for named entity recognition"""

import zipfile as zp
import time

# base path where the data archive is located
PATH_TO_DATA: str = './data/archive.zip'
# file suffixes with data
FILES: dict = {'train': 'train.txt',
               'test': 'test.txt',
               'valid': 'valid.txt'}
# relevant fields to collect
# we want only the token and the named entity for this demonstration
FIELDS: dict = {'token': 0, 'ne': 3}


def read_data(path: str, file: str) -> list:
    """This function opens the zip archive with the data and retrieves data from a file inside
    The function intends to save memory by reading from a zip archive and iterating through each row
    :param path - the str path where the archive is located
    :param file - the the file name with the appropriate data - could be train, test, or valid
    depending on which data set we are looking at for the given task
    :return data - the list containing all the rows data"""
    # initialize data list to store completed sentences
    data = []
    # this list collects sentences as they are being parsed
    sentence = []
    # open archive
    with zp.ZipFile(path) as archive:
        # open file containing data within archive
        with archive.open(file) as f:
            # iterate over rows (lines)
            for row in f:
                if type(row) == bytes:
                    row = row.decode('utf8')
                # these values indicate a sentence ending that should be added to the data
                if row == '-DOCSTART- -X- -X- O\n' or row == '\n':
                    if len(sentence) > 0:
                        # if there is a complete sentence present in the sentence list add to data
                        data.append(sentence)
                        # reset sentence list
                        sentence = []
                # without the newline breaks, the row contains sentence parts that need to be parsed
                else:
                    # split to easily parse elements of the row
                    r = row.split(' ')
                    # add token and named entity data to sentence field
                    sentence.append((r[FIELDS['token']], r[FIELDS['ne']].strip('\n')))
    return data


if __name__ == "__main__":
    tic = time.perf_counter()
    print(read_data(PATH_TO_DATA, FILES['train']))
    toc = time.perf_counter()
    print(toc - tic)
