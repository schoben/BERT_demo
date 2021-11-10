# -*- coding: utf-8 -*-
"""This file loads data from the conll003 dataset and uses a fine-tuned BERT model for named entity recognition"""


import os
import zipfile as zp
import numpy as np
from resources.train_model import train_model
from resources.bert_dataset import BertData
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split


# The base path where the data archive is located.
# This is if we run file separatly
PATH_TO_DATA: str = os.path.join('..', 'data', 'test.zip')
# The file suffixes with data.
FILES: dict = {'train': 'train.txt',
               'test': 'test.txt',
               'valid': 'valid.txt',
               'short': 'ner_short.txt'}

# The relevant fields to collect.
# We want only the token and the named entity for this demonstration.
FIELDS: dict = {'token': 0, 'ne': 3}

# We want tokens for capitalized words now that we are working with named entities
PRETRAINED = 'distilbert-base-cased'

# output directory if running this file alone
OUTPUT_DIR: str = '../results'


def read_data(path: str, file: str) -> (list, list):
    """This function opens the zip archive with the data and retrieves data from a file inside
    The function intends to save memory by reading from a zip archive and iterating through each row
    :param path - the str path where the archive is located
    :param file - the the file name with the appropriate data - could be train, test, or valid
    depending on which data set we are looking at for the given task
    :return data, labels - two lists, one containing all the rows data and another the labels"""
    # initialize data list to store completed sentences
    data = []
    labels = []
    # this list collects sentences as they are being parsed
    sentence = []
    sentence_labels = []
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
                        labels.append(sentence_labels)
                        # reset sentence list
                        sentence = []
                        sentence_labels = []
                # without the newline breaks, the row contains sentence parts that need to be parsed
                else:
                    # split to easily parse elements of the row
                    r = row.split(' ')
                    # add token to and named entity data to respective sentence fields
                    sentence.append(r[FIELDS['token']])
                    sentence_labels.append(r[FIELDS['ne']].strip('\n'))
    # return two separate lists for easier
    return data, labels


def select_subtoken(tags: list, offset_mapping: list, mapping: dict):
    """function to ensure that each label only has one associated token by
    by selecting the first subtoken for tokens with multiple subtokens.
    This is done by encoding labels so the model ignores certain subtokens.
    :param tags: the list of labels
    :param offset_mapping: the list off positions of subtokens for each sentence in the dataset
    start and end position of the subtokens for every token
    :param mapping - a dict that contains named entity classes that converts then to int ids
    :return encoded_labels - an list of labels suitable for use with the list of tokens in the BERT model"""
    # We will leverage the previously made target to int map for get numerical values for each target (label) variable
    labels = [[mapping[tag] for tag in sent] for sent in tags]
    encoded_labels = []
    for row_labels, row_offset in zip(labels, offset_mapping):
        # We are creating an array corresponding the len off the offset map of -100
        # because this is the number used by the model to determine which tokens to ignore.
        row_encode = np.ones(len(row_offset), dtype=int) * -100
        offset_array = np.array(row_offset)

        # We will select only the subtokens in the first position relative to their main tokens to
        # preserve the the consistent shape of the label and feature tensors.
        # All subtokens in the second or third position relative to their main token will be ignored.
        # When first offset for a label is 0 and the second is not, this is the first label
        # for the first subtoken and we want to keep it. Otherwise, we will ignore
        row_encode[(offset_array[:, 0] == 0) & (offset_array[:, 1] != 0)] = row_labels
        encoded_labels.append(row_encode.tolist())
    return encoded_labels


def run_ner(archive: str, file: str, o_dir: str):
    """The main method of the module. This function runs the code that will load, tokenize, and train.
    :param archive - the archive where the txt field is stored
    :param file - the file name of the txt file within the archive
    :param o_dir - the directory to record results"""
    raw_data, raw_labels = read_data(path=archive, file=file)
    # split the data into train and validation sets, ensure data is shuffled
    train_data, val_data, train_label, val_label = train_test_split(raw_data,
                                                                    raw_labels,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    test_size=0.25)
    # tokenize the text, but ensure that the tokenizer understands that words are already separated
    # We will ensure that longer sequences are truncated and shorter ones are padded
    tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    tokens_train = tokenizer(train_data, truncation=True, padding=True,
                             is_split_into_words=True, return_offsets_mapping=True)
    tokens_val = tokenizer(val_data, truncation=True, padding=True,
                           is_split_into_words=True, return_offsets_mapping=True)
    # since we are tokenizing only one token at a time, we need to account for words that the BERT model may not have
    # BERT will split unknown tokens into known sub-tokens, but since we are doing token classification,
    # we need to have only one token for each label, so we will choose only 1 of the sub-tokens
    # we need to start by getting a list of the total unique labels for the tokens
    target_classes = set(tag for row in raw_labels for tag in row)
    # create a 2 way mapping for these classes
    tag_ids = {tag: i for i, tag in enumerate(target_classes)}
    # We are ensuring that only the first subtoken out of a split token gets a label
    train_label = select_subtoken(tags=train_label, offset_mapping=tokens_train['offset_mapping'], mapping=tag_ids)
    val_label = select_subtoken(tags=val_label, offset_mapping=tokens_val['offset_mapping'], mapping=tag_ids)

    # remove offset mapping before creating datasets for pytorch
    tokens_train.pop('offset_mapping')
    tokens_val.pop('offset_mapping')
    # The feature tensor are similar are two dimensional (2589, 148) and contains numerically encoded tokens.
    # This dataset departs from the amazon dataset in that the label tensor has the same dimensions
    # as the feature tensor because there is a label for every token (The Amazon set has 1 label per
    # sequence and so the labels are expressed in 1 dimension). Since we are trying to classify named entities,
    # this makes sense as each entity is a word. Note that the -100 label is there to indicate subtokens and padded
    # values that the model should skip. We are also using a different BERT model -
    # one for token classification to perform this task
    ner_set = BertData(tokens=tokens_train, labels=train_label)
    val_set = BertData(tokens=tokens_val, labels=val_label)
    train_model(data=ner_set, val=val_set, num_labels=len(target_classes), seq=False, o_dir=o_dir)
    print('done')


if __name__ == "__main__":
    # Run this with the appropriate path
    run_ner(archive=PATH_TO_DATA, file=FILES['test'], o_dir=OUTPUT_DIR)
